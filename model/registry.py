"""Single construction point for interchangeable velocity-field backbones."""

from torch import nn

from config.train_config import ModelConfig
from model.velocity_mlp import VelocityMLP
from model.velocity_transformer import VelocityTransformer
from model.velocity_cnn import VelocityCNN
from model.velocity_residual_mlp import VelocityResidualMLP
from model.velocity_unet import VelocityUNet
from model.velocity_global_mlp import VelocityGlobalMLP
from model.velocity_class_conditioned_mlp import VelocityClassConditionedMLP
from model.velocity_modulated_global_mlp import VelocityModulatedGlobalMLP
from model.velocity_adaptive_fourier_mlp import VelocityAdaptiveFourierMLP
from model.velocity_mixture_mlp import VelocityMixtureMLP
from model.velocity_dispersion_guided_mixture_mlp import VelocityDispersionGuidedMixtureMLP
from model.velocity_block_guided_mixture_mlp import VelocityBlockGuidedMixtureMLP
from model.velocity_guided_mixture_mlp import VelocityGuidedMixtureMLP
from model.velocity_oriented_mixture_mlp import VelocityOrientedMixtureMLP
from model.velocity_gaussian_mixture import VelocityGaussianMixture
from model.velocity_correlated_mixture_mlp import VelocityCorrelatedMixtureMLP


MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "mlp": VelocityMLP,
    "transformer": VelocityTransformer,
    "cnn": VelocityCNN,
    "residual_mlp": VelocityResidualMLP,
    "unet": VelocityUNet,
    "global_mlp": VelocityGlobalMLP,
    "class_conditioned_mlp": VelocityClassConditionedMLP,
    "modulated_global_mlp": VelocityModulatedGlobalMLP,
    "adaptive_fourier_mlp": VelocityAdaptiveFourierMLP,
    "mixture_mlp": VelocityMixtureMLP,
    "dispersion_guided_mixture_mlp": VelocityDispersionGuidedMixtureMLP,
    "block_guided_mixture_mlp": VelocityBlockGuidedMixtureMLP,
    "guided_mixture_mlp": VelocityGuidedMixtureMLP,
    "oriented_mixture_mlp": VelocityOrientedMixtureMLP,
    "gaussian_mixture": VelocityGaussianMixture,
    "correlated_mixture_mlp": VelocityCorrelatedMixtureMLP,
}


def build_model(config: ModelConfig) -> nn.Module:
    """Build a registered backbone; legacy checkpoints default to the MLP.

    To add a backbone, implement __init__(ModelConfig) and
    forward(points[B,2], time[B,1], labels[B]) -> velocity[B,2], then register
    the class above. Training, Heun sampling and plotting need no changes.
    """
    config.validate()
    if config.model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {config.model_type!r}; choose from {tuple(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[config.model_type](config)
