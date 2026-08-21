from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    # map_size: int = MISSING
    # minimap_mode: bool = MISSING
    # tag_penalty: float = MISSING
    # max_cycles: int = MISSING
    # extra_features: bool = MISSING
    env_name: str = MISSING
    zone_names: list = MISSING
    use_global_obs: bool = MISSING
    use_redispatching_agent: bool = MISSING
    env_g2op_config: dict = MISSING
    local_rewards: dict = MISSING
    shuffle_chronics: bool = MISSING
    regex_filter_chronics: str = MISSING
    safe_max_rho: float = MISSING
    curtail_margin: float = MISSING
    # PACT-1 settings.  Empty dict (the default) leaves the task byte-identical
    # to the published one, so the blind arm cannot differ by construction.
    pact1: dict = MISSING
    # Dynamic-line-rating severity. TASK physics, applied to every algorithm.
    # 0 = static ratings = stock grid2op; 1 = realistic IEEE 738 derating.
    severity: float = MISSING
