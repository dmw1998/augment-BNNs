"""CPU smoke tests for the ``ebnn`` package.

Consistency checks, not paper numbers: models build and backpropagate, the C4 projection
is idempotent and invariant, the vectorised ``enforce`` operators match the reference ones,
P4CNN / orbit-averaged samples are exactly invariant, the single-pass validation matches
the reference metrics, and the NCT-CRC-HE preprocessing works on a synthetic zip.

Run with:  python -m pytest tests/test_smoke.py   (or)   python tests/test_smoke.py
"""

import copy
import io
import os
import sys
import tempfile
import types
import zipfile
from unittest import mock

# Allow the package (whose __init__ imports ebnn.data -> torchvision) to import
# even when torchvision is absent; the data path is not exercised here.
try:
    import torchvision  # noqa: F401
except Exception:  # pragma: no cover - test-environment shim only
    stub = types.ModuleType("torchvision")
    stub.datasets = types.SimpleNamespace(MNIST=object, FashionMNIST=object, CIFAR10=object, CIFAR100=object)
    transforms = types.ModuleType("torchvision.transforms")

    class _T:
        @staticmethod
        def Compose(*a, **k):
            return None

        @staticmethod
        def ToTensor(*a, **k):
            return None

        @staticmethod
        def Normalize(*a, **k):
            return None

    transforms.Compose = _T.Compose
    transforms.ToTensor = _T.ToTensor
    transforms.Normalize = _T.Normalize
    transforms.functional = types.SimpleNamespace(rotate=lambda img, angle: img)
    stub.transforms = transforms
    sys.modules["torchvision"] = stub
    sys.modules["torchvision.transforms"] = transforms

import numpy as np  # noqa: E402
import torch  # noqa: E402

import ebnn  # noqa: E402
from ebnn.data import get_num_classes  # noqa: E402
from ebnn.histology import NCTCRCHE  # noqa: E402
from ebnn.orbit import OrbitAveragedBNN  # noqa: E402
from ebnn.p4cnn import BayesianP4CNN  # noqa: E402
from ebnn.models import (  # noqa: E402
    BayesianCNN,
    BayesianCNNConfigurable,
    BayesianFullyConvCNN,
    BayesianGroupConvCNN,
)
from ebnn.symmetrize import (  # noqa: E402
    apply_c4_enforcement,
    apply_one_shot_c4_posterior_average,
    apply_one_shot_c4_projection,
    build_optimizer,
    expand_small_model_to_gcnn,
    initialize_gaussian_posterior_scale,
    initialize_posterior_mean,
)

torch.manual_seed(0)


def _forward_backward(model, x, family="gaussian"):
    out = model(x, return_kl=True)
    logits, kl = out[0], out[-1]
    loss = logits.float().pow(2).mean() + kl
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite"
    return logits


def test_theorem_cnn():
    model = BayesianCNN(1, 10, conv_channels=[8, 16], input_size=(28, 28))
    logits = _forward_backward(model, torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


def test_configurable_families():
    for family in ("gaussian", "laplace", "lognormal"):
        model = BayesianCNNConfigurable(
            in_channels=1,
            num_classes=10,
            family=family,
            conv_channels=[8, 16],
            input_size=(28, 28),
        )
        logits = _forward_backward(model, torch.randn(4, 1, 28, 28))
        assert logits.shape == (4, 10), family


def test_fullyconv_forward():
    model = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    logits = _forward_backward(model, torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


def test_projection_is_idempotent():
    # Channel counts must be divisible by 4 for the C4 projection.
    model = BayesianFullyConvCNN(1, 12, conv_channels=[8, 16])
    apply_one_shot_c4_projection(model)
    w1 = model.conv_layers[1].weight_mu.detach().clone()
    apply_one_shot_c4_projection(model)
    w2 = model.conv_layers[1].weight_mu.detach().clone()
    assert torch.allclose(w1, w2, atol=1e-5), "projection should be idempotent on the mean"


def test_posterior_average_runs():
    model = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    for method in ("geometric", "arithmetic"):
        apply_one_shot_c4_posterior_average(model, method=method)
    logits = model(torch.randn(2, 1, 28, 28), return_kl=False)
    assert logits.shape == (2, 10)


def test_gcnn_expansion_runs():
    small = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    initialize_posterior_mean(small, posterior_init="kaiming")
    initialize_gaussian_posterior_scale(small, rho_init=-3.0)
    gcnn = BayesianGroupConvCNN(1, 10, base_channels=[8, 16], group_size=4)
    expand_small_model_to_gcnn(small, gcnn, group_size=4)
    logits = gcnn(torch.randn(2, 1, 28, 28), return_kl=False)
    assert logits.shape == (2, 10)


def test_optimizer_builders():
    model = BayesianFullyConvCNN(1, 10, conv_channels=[8, 16])
    adamw = build_optimizer(model, optimizer="adamw", lr=1e-3)
    sgd = build_optimizer(model, optimizer="sgd", lr=1e-2)
    assert len(adamw.param_groups) == 1
    assert len(sgd.param_groups) == 2  # base + rho groups


def _randomize_posterior(model):
    """Kaiming means, random biases and non-constant scales, so every projection branch is exercised."""
    initialize_posterior_mean(model, posterior_init="kaiming")
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("rho"):
                p.normal_(-3.0, 0.5)
            elif name.endswith("bias_mu"):
                p.normal_(0.0, 0.1)


def _max_rotation_gap(model, x):
    """Max |logits(rot x) - logits(x)| over the C4 rotations, with the posterior noise switched off."""
    with mock.patch.object(torch, "randn_like", torch.zeros_like), torch.no_grad():
        ref = model(x, return_kl=False)
        rotated = [model(torch.rot90(x, k, dims=[-2, -1]), return_kl=False) for k in (1, 2, 3)]
    return max((r - ref).abs().max().item() for r in rotated)


def test_projection_rgb_is_invariant():
    # The lifting-layer projection must constrain every input channel, not only channel 0.
    model = BayesianFullyConvCNN(3, 9, conv_channels=[8, 16])
    _randomize_posterior(model)
    apply_one_shot_c4_projection(model)
    assert _max_rotation_gap(model, torch.randn(2, 3, 16, 16)) < 1e-4


def test_fast_projection_matches_reference():
    model = BayesianFullyConvCNN(3, 9, conv_channels=[8, 16])
    _randomize_posterior(model)
    ref, fast = copy.deepcopy(model), copy.deepcopy(model)
    apply_one_shot_c4_projection(ref)
    apply_c4_enforcement(fast, op="proj")
    for (name, p_ref), p_fast in zip(ref.named_parameters(), fast.parameters()):
        if name == "classifier.weight_rho":
            continue  # the enforce projection additionally averages the classifier scale
        assert torch.allclose(p_ref, p_fast, atol=1e-6), name
    sigma = torch.nn.functional.softplus(fast.classifier.weight_rho).view(9, -1, 4)
    assert torch.allclose(sigma, sigma.mean(dim=2, keepdim=True).expand_as(sigma), atol=1e-6)


def test_enforce_rescale_and_projected_steps():
    for op in ("proj", "avg"):
        model = BayesianFullyConvCNN(3, 9, conv_channels=[8, 16])
        initialize_posterior_mean(model, posterior_init="kaiming")
        initialize_gaussian_posterior_scale(model, rho_init=-3.0)
        norms = {n: p.norm().item() for n, p in model.named_parameters() if n.endswith("weight_mu")}
        factors = apply_c4_enforcement(model, op=op, rescale=True)
        assert factors["conv_layers.0"][0] > 1.2, (op, factors)  # symmetrisation shrank the init ...
        for n, p in model.named_parameters():
            if n.endswith("weight_mu"):
                assert abs(p.norm().item() - norms[n]) < 1e-4 * norms[n], (op, n)  # ... and is undone
        x = torch.randn(2, 3, 16, 16)
        assert _max_rotation_gap(model, x) < 1e-4, op

        state = copy.deepcopy(model.state_dict())  # idempotent on the subspace
        apply_c4_enforcement(model, op=op)
        for k, v in model.state_dict().items():
            assert torch.allclose(v, state[k], atol=1e-5), (op, k)

        with torch.no_grad():  # a non-equivariant "optimizer step", then re-enforce
            for p in model.parameters():
                p.add_(0.05 * torch.randn_like(p))
        apply_c4_enforcement(model, op=op)
        assert _max_rotation_gap(model, x) < 1e-4, op


def test_p4cnn_samples_are_invariant():
    model = BayesianP4CNN(3, 9, base_channels=[4, 8, 8])
    initialize_posterior_mean(model, posterior_init="kaiming")
    initialize_gaussian_posterior_scale(model, rho_init=-1.0)  # large posterior noise
    _forward_backward(model, torch.randn(2, 3, 16, 16))
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        torch.manual_seed(1)
        ref = model(x, return_kl=False)
        for k in (1, 2, 3):  # same posterior sample (same noise), rotated input
            torch.manual_seed(1)
            gap = (model(torch.rot90(x, k, dims=[-2, -1]), return_kl=False) - ref).abs().max().item()
            assert gap < 1e-5 * ref.abs().max().item(), (k, gap)
    # Posterior on the free filters only: lifting [C, C_in, 3, 3], group [C, C', 4, 3, 3].
    assert model.conv_layers[0].weight_mu.shape == (4, 3, 3, 3)
    assert model.conv_layers[1].weight_mu.shape == (8, 4, 4, 3, 3)
    build_optimizer(model, "sgd", lr=1e-2, weight_decay=0.0, nesterov=True).step()


def test_orbit_averaged_samples_are_invariant():
    for lik in ("prob", "logit"):
        base = BayesianFullyConvCNN(in_channels=3, num_classes=9, conv_channels=[8, 8])
        initialize_posterior_mean(base, posterior_init="kaiming")
        initialize_gaussian_posterior_scale(base, rho_init=-1.0)  # large posterior noise
        model = OrbitAveragedBNN(base, likelihood=lik)
        _forward_backward(model, torch.randn(2, 3, 16, 16))
        x = torch.randn(2, 3, 16, 16)
        with torch.no_grad():
            torch.manual_seed(1)
            ref = model(x, return_kl=False)
            # softmax of the returned logits is the orbit-averaged predictive (same sample)
            torch.manual_seed(1)
            orbit_logits = base(torch.cat([torch.rot90(x, r, dims=(2, 3)) for r in range(4)]), return_kl=False)
            orbit_logits = orbit_logits.view(4, 2, -1)
            target = orbit_logits.softmax(-1).mean(0) if lik == "prob" else orbit_logits.mean(0).softmax(-1)
            assert torch.allclose(ref.softmax(-1), target, atol=1e-6), lik
            for k in (1, 2, 3):  # same posterior sample (same noise), rotated input
                torch.manual_seed(1)
                gap = (model(torch.rot90(x, k, dims=(2, 3)), return_kl=False) - ref).abs().max().item()
                assert gap < 1e-5 * ref.abs().max().item(), (lik, k, gap)
        build_optimizer(model, "sgd", lr=1e-2, weight_decay=0.0, nesterov=True).step()


def test_single_pass_validation_matches_reference_metrics():
    # Deterministic network, so the single pass and the separate reference passes see
    # identical predictive distributions and must agree up to float error.
    from ebnn.metrics import (
        compute_orbits_same_pred_mc,
        compute_symmetric_kl_divergence_mc,
        eval_pass,
        evaluate_c4_orbit,
        run_validation,
    )

    class _Det(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 5))

        def forward(self, x, return_kl=True):
            out = 3.0 * self.net(x)
            return (out, torch.zeros(())) if return_kl else out

    model = _Det()
    x = torch.randn(23, 3, 8, 8)
    y = torch.randint(0, 5, (23,))
    equiv_loader = [
        (*[torch.rot90(x[i : i + 7], r, dims=(2, 3)) for r in range(4)], y[i : i + 7]) for i in range(0, 23, 7)
    ]
    test_loader = [(x[i : i + 7], y[i : i + 7]) for i in range(0, 23, 7)]

    res = evaluate_c4_orbit(model, equiv_loader, "cpu", eval_samples=2)
    acc_c4, loss_c4 = eval_pass(model, test_loader, equiv_loader, "cpu", 2, eval_source="equiv_all")
    acc_orig, loss_orig = eval_pass(model, test_loader, equiv_loader, "cpu", 2, eval_source="test")
    kl = compute_symmetric_kl_divergence_mc(model, equiv_loader, "cpu", 2)
    assert abs(res["acc_c4"] - acc_c4) < 1e-9 and abs(res["acc_orig"] - acc_orig) < 1e-9
    assert abs(res["loss_c4"] - loss_c4) < 1e-5 and abs(res["loss_orig"] - loss_orig) < 1e-5
    assert abs(res["osp"] - compute_orbits_same_pred_mc(model, equiv_loader, "cpu", 2)) < 1e-9
    assert abs(res["kl_mean"] - kl["mean"]) < 1e-5 and abs(res["kl_std"] - kl["std"]) < 1e-5
    for source, key in (("equiv_all", "c4"), ("test", "orig")):
        metrics, dist = run_validation(model, test_loader, equiv_loader, "cpu", 2, source)
        assert metrics["val/accuracy"] == res[f"acc_{key}"]
        assert len(dist["pred_labels"]) == (4 * 23 if key == "c4" else 23)


def test_nct_crc_he_from_synthetic_zip():
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - test-environment shim only
        print("SKIP  test_nct_crc_he_from_synthetic_zip (Pillow not installed)")
        return
    assert get_num_classes("NCT_CRC_HE") == 9
    with tempfile.TemporaryDirectory() as root:
        raw = os.path.join(root, "NCT_CRC_HE", "raw")
        os.makedirs(raw)
        zip_path = os.path.join(raw, "CRC-VAL-HE-7K.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for value, cls in enumerate(("TUM", "ADI", "NORM")):
                for j in range(2):
                    buf = io.BytesIO()
                    Image.fromarray(np.full((224, 224, 3), 10 * value + j, np.uint8)).save(buf, format="TIFF")
                    zf.writestr(f"CRC-VAL-HE-7K/{cls}/{cls}-{j}.tif", buf.getvalue())
        ds = NCTCRCHE(root, train=False, image_size=32)  # decodes + resizes once
        assert ds.data.shape == (6, 32, 32, 3) and ds.data.dtype == np.uint8
        assert ds.targets == [0, 0, 6, 6, 8, 8]  # sorted folders ADI, NORM, TUM
        assert int(ds.data[0].min()) == int(ds.data[0].max()) == 10  # ADI-0
        img, label = ds[5]
        assert img.size == (32, 32) and label == 8
        os.remove(zip_path)  # later runs only need the processed arrays
        again = NCTCRCHE(root, train=False, image_size=32)
        assert again.targets == ds.targets and np.array_equal(again.data, ds.data)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} smoke tests passed. Public API exports: {len(ebnn.__all__)} names.")


if __name__ == "__main__":
    _run_all()
