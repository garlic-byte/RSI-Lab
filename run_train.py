"""Train a selectable conditional velocity model on noisy point distributions."""

import argparse
from dataclasses import fields

from config.train_config import ModelConfig, TrainConfig
from train.trainer import run_training
from model.registry import MODEL_REGISTRY


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
    parser.add_argument("--model", choices=tuple(MODEL_REGISTRY), default="mlp")
    parser.add_argument("--transformer-dim", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    args = vars(parser.parse_args())
    model_config = ModelConfig(hidden_dim=args.pop("hidden_dim"), num_layers=args.pop("num_layers"),
                               num_classes=args["num_classes"], model_type=args.pop("model"),
                               transformer_dim=args.pop("transformer_dim"), num_heads=args.pop("num_heads"))
    try:
        model_config.validate()
    except ValueError as error:
        parser.error(str(error))
    run_training(TrainConfig(**args), model_config)


if __name__ == "__main__":
    main()
