"""Public-information student distilled from a complete-wall policy teacher."""
import copy

import numpy as np
import torch
from torch import nn

from .action_scheme import ACTION_N
from .observation import LAYOUT, TILE_KINDS
from .oracle_wall import OracleWallPolicy, ORACLE_OBS_DIM, FLOAT_MIN, DISTANCE_CLASSES


def public_observation_from_oracle(observation):
    """Remove all ordered-wall positions; retain public planes and legal bits."""
    if observation.shape[-1] != ORACLE_OBS_DIM:
        raise ValueError('Expected a complete-wall observation')
    pieces = (observation[..., :LAYOUT.tile_features], observation[..., -ACTION_N:])
    if torch.is_tensor(observation):
        return torch.cat(pieces, dim=-1)
    return np.concatenate(pieces, axis=-1)


class PublicStudentPolicy(OracleWallPolicy):
    observation_mode = 'public_v2'

    def __init__(self, *, model_config):
        super().__init__(model_config=model_config)
        old = self.tiles[0]
        self.tiles[0] = nn.Conv1d(LAYOUT.tile_channels, old.out_channels, old.kernel_size,
                                 stride=old.stride, padding=old.padding, bias=False)

    def forward(self, observation):
        if observation.shape[-1] != LAYOUT.flat_dim:
            raise ValueError(f'Public student requires exactly {LAYOUT.flat_dim} features')
        leading = observation.shape[:-1]
        flat = observation.reshape(-1, LAYOUT.flat_dim)
        planes = flat[:, :LAYOUT.tile_features].reshape(-1, LAYOUT.tile_channels, TILE_KINDS)
        if getattr(self, 'activation_checkpoint', False) and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint_sequential
            encoded = checkpoint_sequential(self.tiles, min(8, len(self.tiles)), planes,
                                            use_reentrant=False)
        else:
            encoded = self.tiles(planes)
        latent = self.projection(encoded)
        logits = self.pi(latent) + torch.clamp(torch.log(flat[:, -ACTION_N:]), min=FLOAT_MIN)
        distance = self.distance_head(latent).reshape(*leading, 3, DISTANCE_CLASSES)
        ready = distance.argmax(-1) == 0
        win = self.win_head(latent).reshape(*leading, 3)
        fan = self.fan_exponent_head(latent).reshape(*leading, 3)
        return dict(policy_logits=logits.reshape(*leading, ACTION_N), distance_logits=distance,
                    ready_mask=ready, win_logits=win,
                    win_probability=torch.where(ready, win.sigmoid(), 0.0),
                    fan_exponent=fan,
                    masked_fan_exponent=torch.where(ready, fan.clamp(0.0, 9.0), 0.0))


def initialize_public_bundle(teacher_bundle, teacher_sha256):
    """Copy compatible weights, slice the public stem, and start fresh Adam state."""
    if teacher_bundle.get('oracle_wall_version') != 1 or teacher_bundle.get('observation_dim') != ORACLE_OBS_DIM:
        raise ValueError('Teacher must be the original complete-wall model')
    student = PublicStudentPolicy(model_config=teacher_bundle['model_config'])
    weights = {name: value.detach().cpu().clone() for name, value in teacher_bundle['actor'].items()}
    weights['tiles.0.weight'] = weights['tiles.0.weight'][:, :LAYOUT.tile_channels].clone()
    student.load_state_dict(weights, strict=True)
    bundle = dict(public_student_version=1, observation_version=2,
                  observation_dim=LAYOUT.flat_dim, model_config=copy.deepcopy(teacher_bundle['model_config']),
                  actor=weights, teacher_sha256=teacher_sha256,
                  removed_features='all 136 ordered-wall tile planes; no opponent concealed hand inputs',
                  retained_features='208 public planes and 41 legal-action bits',
                  initialization='teacher weights; input stem sliced to public channels; new Adam state')
    return student, bundle


def load_public_bundle(bundle):
    if bundle.get('public_student_version') != 1 or bundle.get('observation_dim') != LAYOUT.flat_dim:
        raise ValueError('Not a compatible public-information student')
    model = PublicStudentPolicy(model_config=bundle['model_config'])
    model.load_state_dict(bundle['actor'], strict=True)
    return model.eval()
