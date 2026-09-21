"""杭州麻将环境适配；规则引擎由同目录的 hangzhou-mahjong 子模块提供。"""
from dataclasses import asdict

ENV_ID = "HZMahjong"


def make_env(env_config=None):
    from .mahjong_env import make_mahjong_env
    return make_mahjong_env(env_config)


def get_module_class(algorithm):
    from .mahjong_module import get_module_class as select
    return select(algorithm)


def environment_config(settings):
    values = asdict(settings.environment)
    for key in ("vectorize_mode", "sample_timeout_s", "batch_mode"):
        values.pop(key)
    return values


def add_cli_arguments(parser):
    parser.add_argument("--opponent", default=None, help="麻将对手池：rule / greedy / random / rule_aggressive / rule_safe")


def local_overrides(args, settings):
    result = {"opponent": args.opponent or settings.environment.opponent, "max_episode_steps": args.episode_length}
    if args.opponent:
        result['opponent_pool'] = None
    return result


def evaluation_config(settings, checkpoint):
    return {**environment_config(settings), 'opponent': 'rule', 'opponent_pool': None,
            'unique_episode_seeds': False}
