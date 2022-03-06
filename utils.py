import yaml


def read_config(cfg_path: str = "config.yaml") -> dict:
    algo_config = None

    if cfg_path is not None:
        with open(cfg_path, "r") as f:
            algo_config = yaml.load(f, Loader=yaml.FullLoader)

    return algo_config
