"""Opt-in perfect-information 41-action actor with three opponent prediction heads.

The 136 wall slots are in their original, *ordered* deal/draw sequence.  This
is an oracle experiment: inference requires the complete wall, including
already dealt tiles.  It must never be loaded by the public platform adapter.
"""

import copy

import torch
from torch import nn
import torch.nn.functional as F

from .action_scheme import ACTION_N
from .mortal_blocks import MortalResidualBlock
from .observation import LAYOUT, TILE_KINDS


WALL_SLOTS = 136
WALL_FEATURES = WALL_SLOTS * TILE_KINDS
ORACLE_CHANNELS = LAYOUT.tile_channels + WALL_SLOTS
ORACLE_OBS_DIM = LAYOUT.flat_dim + WALL_FEATURES
DISTANCE_CLASSES = 14  # shanten 0..13; last bin also clips larger gaps
FLOAT_MIN = -3.4e38  # Same mask sentinel used by the trained 41-action actor.


class StandalonePolicyHead(nn.Module):
    """State-dict-compatible categorical head; inference does not need RLlib."""

    def __init__(self, latent):
        super().__init__()
        self.register_buffer('log_std_clip_param_const', torch.zeros(1))
        self.net = nn.Module()
        self.net.mlp = nn.Sequential(nn.Linear(latent, 256), nn.Tanh(),
                                    nn.Linear(256, ACTION_N))

    def forward(self, latent):
        return self.net.mlp(latent)


class StandaloneParentPolicy(nn.Module):
    """Read the existing 41-action actor checkpoint without importing RLlib."""

    def __init__(self, model_config):
        super().__init__()
        if (model_config.get('encoder_type') != 'mortal_v4' or
                model_config.get('observation_canonicalization', 'none') != 'none'):
            raise ValueError('Expected a non-canonical Mortal-v4 parent')
        channels = int(model_config['resnet_channels'])
        groups = int(model_config['resnet_groups'])
        blocks = int(model_config['resnet_blocks'])
        latent = int(model_config['latent_dim'])
        self.encoder = nn.Module()
        self.encoder.actor_encoder = nn.Module()
        self.encoder.actor_encoder.tiles = nn.Sequential(
            nn.Conv1d(LAYOUT.tile_channels, channels, 3, padding=1, bias=False),
            *(MortalResidualBlock(channels, groups) for _ in range(blocks)),
            nn.GroupNorm(groups, channels, eps=1e-3), nn.Mish(),
            nn.Conv1d(channels, 32, 3, padding=1), nn.Mish(), nn.Flatten(1),
        )
        self.encoder.actor_encoder.projection = nn.Sequential(
            nn.Linear(32 * TILE_KINDS, latent), nn.Mish())
        self.pi = StandalonePolicyHead(latent)
        self.canonical_observation = False

    def forward(self, observation):
        if observation.shape[-1] != LAYOUT.flat_dim:
            raise ValueError(f'Expected {LAYOUT.flat_dim} public features')
        leading = observation.shape[:-1]
        flat = observation.reshape(-1, LAYOUT.flat_dim)
        planes = flat[:, :-ACTION_N].reshape(-1, LAYOUT.tile_channels, TILE_KINDS)
        trunk = self.encoder.actor_encoder
        latent = trunk.projection(trunk.tiles(planes))
        logits = self.pi(latent) + torch.clamp(torch.log(flat[:, -ACTION_N:]), min=FLOAT_MIN)
        return logits.reshape(*leading, ACTION_N)


def encode_wall_observation(public_observation, wall):
    """Append 136 ordered one-hot tiles before the unchanged 41 legal bits."""
    obs = torch.as_tensor(public_observation)
    tiles = torch.as_tensor(wall, device=obs.device)
    if obs.shape[-1] != LAYOUT.flat_dim or tiles.shape != obs.shape[:-1] + (WALL_SLOTS,):
        raise ValueError('Expected public [*,7113] and ordered wall [*,136]')
    one_hot = F.one_hot(tiles.long(), TILE_KINDS).reshape(*obs.shape[:-1], WALL_FEATURES)
    return torch.cat((obs[..., :-ACTION_N], one_hot.to(obs.dtype),
                      obs[..., -ACTION_N:]), dim=-1)


class OracleWallPolicy(nn.Module):
    """Mortal-v4 trunk + 41-action policy and three parallel FC prediction heads.

    Head outputs: [B,3,14] shanten classes, [B,3] next-draw win logits,
    [B,3] next-draw fan exponent (log2 of engine fan). Relative opponents are
    seats +1,+2,+3. Training masks the last two heads for non-ready hands.
    """

    def __init__(self, parent_policy=None, *, model_config=None):
        super().__init__()
        if (parent_policy is None) == (model_config is None):
            raise ValueError('Supply exactly one of parent_policy or model_config')
        if parent_policy is not None:
            if parent_policy.canonical_observation:
                raise ValueError('Oracle wall model currently requires non-canonical parent')
            source = parent_policy.encoder.actor_encoder
            self.tiles = copy.deepcopy(source.tiles)
            old = self.tiles[0]
            if not isinstance(old, nn.Conv1d) or old.in_channels != LAYOUT.tile_channels:
                raise ValueError('Expected a 208-channel Mortal-v4 parent stem')
            stem = nn.Conv1d(ORACLE_CHANNELS, old.out_channels, old.kernel_size,
                             stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                stem.weight.zero_()
                stem.weight[:, :LAYOUT.tile_channels].copy_(old.weight)
            self.tiles[0] = stem
            self.projection = copy.deepcopy(source.projection)
            self.pi = copy.deepcopy(parent_policy.pi)
            latent = self.projection[0].out_features
        else:
            if (model_config.get('encoder_type') != 'mortal_v4' or
                    model_config.get('observation_canonicalization', 'none') != 'none'):
                raise ValueError('Expected a non-canonical Mortal-v4 model')
            channels = int(model_config['resnet_channels'])
            groups = int(model_config['resnet_groups'])
            blocks = int(model_config['resnet_blocks'])
            latent = int(model_config['latent_dim'])
            self.tiles = nn.Sequential(
                nn.Conv1d(ORACLE_CHANNELS, channels, 3, padding=1, bias=False),
                *(MortalResidualBlock(channels, groups) for _ in range(blocks)),
                nn.GroupNorm(groups, channels, eps=1e-3), nn.Mish(),
                nn.Conv1d(channels, 32, 3, padding=1), nn.Mish(), nn.Flatten(1),
            )
            self.projection = nn.Sequential(nn.Linear(32 * TILE_KINDS, latent), nn.Mish())
            self.pi = StandalonePolicyHead(latent)
        self.distance_head = nn.Linear(latent, 3 * DISTANCE_CLASSES)
        self.win_head = nn.Linear(latent, 3)
        self.fan_exponent_head = nn.Linear(latent, 3)

    def forward(self, observation):
        if observation.shape[-1] != ORACLE_OBS_DIM:
            raise ValueError(f'Oracle input must have {ORACLE_OBS_DIM} features')
        leading = observation.shape[:-1]
        flat = observation.reshape(-1, ORACLE_OBS_DIM)
        planes = flat[:, :-ACTION_N].reshape(-1, ORACLE_CHANNELS, TILE_KINDS)
        if getattr(self, 'activation_checkpoint', False) and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint_sequential
            encoded = checkpoint_sequential(self.tiles, 8, planes, use_reentrant=False)
        else:
            encoded = self.tiles(planes)
        latent = self.projection(encoded)
        mask = flat[:, -ACTION_N:]
        raw_logits = self.pi(latent)
        logits = raw_logits + torch.clamp(torch.log(mask), min=FLOAT_MIN)
        distance_logits = self.distance_head(latent).reshape(*leading, 3, DISTANCE_CLASSES)
        win_logits = self.win_head(latent).reshape(*leading, 3)
        fan_exponent = self.fan_exponent_head(latent).reshape(*leading, 3)
        predicted_ready = distance_logits.argmax(-1) == 0
        return {
            'policy_logits': logits.reshape(*leading, ACTION_N),
            'distance_logits': distance_logits,
            'ready_mask': predicted_ready,
            'win_logits': win_logits,
            'win_probability': torch.where(predicted_ready, win_logits.sigmoid(), 0.0),
            'fan_exponent': fan_exponent,
            'masked_fan_exponent': torch.where(predicted_ready,
                                               fan_exponent.clamp(0.0, 9.0), 0.0),
        }

    def act(self, observation):
        return self(observation)['policy_logits'].argmax(-1)


def load_parent(bundle):
    """Construct a source 41-action BC actor without changing its checkpoint."""
    if bundle.get('observation_version') != 2 or bundle['model'].get('encoder_type') != 'mortal_v4':
        raise ValueError('Expected a v2 Mortal-v4 41-action actor')
    parent = StandaloneParentPolicy(bundle['model'])
    parent.load_state_dict(bundle['actor'], strict=True)
    return parent


def load_oracle_bundle(bundle):
    """Load a saved oracle actor; no parent checkpoint is needed at inference."""
    if bundle.get('small_selfplay_version') == 1:
        from .small_selfplay import load_small_bundle
        return load_small_bundle(bundle)
    if bundle.get('public_student_version') == 1:
        from .public_student import load_public_bundle
        return load_public_bundle(bundle)
    if bundle.get('oracle_wall_version') != 1 or bundle.get('observation_dim') != ORACLE_OBS_DIM:
        raise ValueError('Not a compatible complete-wall actor bundle')
    actor = OracleWallPolicy(model_config=bundle['model_config'])
    actor.load_state_dict(bundle['actor'], strict=True)
    return actor.eval()
