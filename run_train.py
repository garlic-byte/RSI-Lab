"""Train the conditional MLP on configurable analytic noisy point distributions."""

import argparse
from dataclasses import fields

from config.train_config import ModelConfig, TrainConfig
from train.trainer import run_training


def main() -> None:
    """Expose all training settings as standard argparse flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    for field in fields(defaults):
        value = getattr(defaults, field.name)
        flag = "--" + field.name.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=value,
                                help=("Save generated galleries every --save-every steps" if field.name == "save_progress"
                                      else "Animate particles during final inference"))
        else:
            parser.add_argument(flag, type=type(value), default=value)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=4)
    args = vars(parser.parse_args())
    model_config = ModelConfig(hidden_dim=args.pop("hidden_dim"), num_layers=args.pop("num_layers"), num_classes=args["num_classes"])
    if model_config.hidden_dim <= 0 or model_config.num_layers <= 0:
        parser.error("hidden-dim and num-layers must be positive")
    run_training(TrainConfig(**args), model_config)


if __name__ == "__main__":
    main()
