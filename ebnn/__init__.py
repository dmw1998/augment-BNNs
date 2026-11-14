"""Equivariance and augmentation for Bayesian neural networks (C4 / FashionMNIST).

Public API for the experiment scripts under ``scripts/``.
"""

from ebnn.data import (
    C4AugmentedDataset,
    EquivarianceEvalSet,
    RandomC4AugmentedDataset,
    RepeatedDataset,
    get_base_transform,
    get_loaders,
    get_num_classes,
)
from ebnn.histology import NCTCRCHE
from ebnn.layers import (
    BayesianConv2dGaussian,
    BayesianConv2dLaplace,
    BayesianConv2dLogNormal,
    BayesianLinearGaussian,
    BayesianLinearLaplace,
    BayesianLinearLogNormal,
)
from ebnn.metrics import (
    compute_equivariance_defect,
    compute_equivariance_defect_K_fold,
    compute_equivariance_defect_T_sweep,
    compute_orbits_same_pred,
    compute_orbits_same_pred_mc,
    compute_symmetric_kl_divergence,
    compute_symmetric_kl_divergence_mc,
    eval_pass,
    evaluate_c4_orbit,
    evaluate_classification_accuracy,
    predict_probs_mc,
    run_validation,
)
from ebnn.models import (
    BayesianCNN,
    BayesianCNNConfigurable,
    BayesianFullyConvCNN,
    BayesianGroupConvCNN,
)
from ebnn.orbit import OrbitAveragedBNN, c4_orbit
from ebnn.p4cnn import BayesianP4CNN
from ebnn.symmetrize import (
    apply_c4_enforcement,
    apply_one_shot_c4_posterior_average,
    apply_one_shot_c4_projection,
    build_optimizer,
    clear_optimizer_state,
    describe_optimizer,
    expand_small_model_to_gcnn,
    initialize_gaussian_posterior_scale,
    initialize_posterior_mean,
    rotate_kernel,
    rotate_kernel_c4,
)

__all__ = [
    "BayesianCNN",
    "BayesianCNNConfigurable",
    "BayesianConv2dGaussian",
    "BayesianConv2dLaplace",
    "BayesianConv2dLogNormal",
    "BayesianFullyConvCNN",
    "BayesianGroupConvCNN",
    "BayesianP4CNN",
    "OrbitAveragedBNN",
    "c4_orbit",
    "BayesianLinearGaussian",
    "BayesianLinearLaplace",
    "BayesianLinearLogNormal",
    "C4AugmentedDataset",
    "EquivarianceEvalSet",
    "NCTCRCHE",
    "RandomC4AugmentedDataset",
    "RepeatedDataset",
    "apply_c4_enforcement",
    "apply_one_shot_c4_posterior_average",
    "apply_one_shot_c4_projection",
    "build_optimizer",
    "clear_optimizer_state",
    "compute_equivariance_defect",
    "compute_equivariance_defect_K_fold",
    "compute_equivariance_defect_T_sweep",
    "compute_orbits_same_pred",
    "compute_orbits_same_pred_mc",
    "compute_symmetric_kl_divergence",
    "compute_symmetric_kl_divergence_mc",
    "describe_optimizer",
    "eval_pass",
    "evaluate_c4_orbit",
    "evaluate_classification_accuracy",
    "expand_small_model_to_gcnn",
    "get_base_transform",
    "get_loaders",
    "get_num_classes",
    "initialize_gaussian_posterior_scale",
    "initialize_posterior_mean",
    "predict_probs_mc",
    "rotate_kernel",
    "rotate_kernel_c4",
    "run_validation",
]
