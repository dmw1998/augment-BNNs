"""Datasets and data loaders with C4 (90-degree rotation) augmentation.

``get_loaders`` returns three loaders:

* ``train_loader``  -- training data according to ``augmentation_mode``
  (``"without"`` / ``"full"`` / ``"random"``).
* ``test_loader``   -- original test data (no augmentation).
* ``equiv_loader``  -- test data in equivariance-evaluation format, yielding the
  full C4 orbit ``(img_0, img_90, img_180, img_270, label)`` per sample.

The number of DataLoader workers is configurable (it does not affect results for
the deterministic ``"without"`` / ``"full"`` modes).

Datasets: MNIST / FashionMNIST / CIFAR10 / CIFAR100 (torchvision) and the
NCT-CRC-HE colorectal histology set (:mod:`ebnn.histology`, ``"NCT_CRC_HE"``).
"""

import hashlib
import os
import tempfile

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

from ebnn.histology import DEFAULT_IMAGE_SIZE, NCTCRCHE


def _hash_int_list(values):
    h = hashlib.sha1()
    for v in values:
        h.update(str(int(v)).encode("ascii"))
        h.update(b",")
    return h.hexdigest()[:12]


def _subset_signature(dataset):
    if isinstance(dataset, torch.utils.data.Subset):
        return f"subset-{_hash_int_list(dataset.indices)}"
    return "full"


def _build_cache_path(cache_dir, prefix, dataset_name, suffix):
    os.makedirs(cache_dir, exist_ok=True)
    filename = f"{prefix}_{dataset_name}_{suffix}.pt"
    return os.path.join(cache_dir, filename)


def _atomic_torch_save(payload, path):
    """``torch.save`` via a temp file + rename, so concurrent jobs never load a half-written cache."""
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def get_base_transform(dataset_name="MNIST"):
    if dataset_name in ("MNIST", "FashionMNIST"):
        return T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
    if dataset_name in ("CIFAR10", "CIFAR100", "NCT_CRC_HE"):
        return T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    raise ValueError(f"Unknown dataset: {dataset_name}")


class C4AugmentedDataset(Dataset):
    """Full C4 augmentation: yields ``(rotated_image, label, angle)`` for every orbit element."""

    def __init__(self, base_dataset, angles=(0, 90, 180, 270), cache_path=None):
        self.base_dataset = base_dataset
        self.angles = tuple(angles)

        loaded = False
        if cache_path is not None and os.path.exists(cache_path):
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.images = payload["images"]
            self.labels = payload["labels"]
            self.angles_tensor = payload["angles"]
            loaded = True

        if not loaded:
            images, labels, angles_out = [], [], []
            for idx in range(len(base_dataset)):
                img, label = base_dataset[idx]
                for angle in self.angles:
                    images.append(T.functional.rotate(img, angle))
                    labels.append(int(label))
                    angles_out.append(int(angle))

            self.images = torch.stack(images, dim=0)
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.angles_tensor = torch.tensor(angles_out, dtype=torch.long)

            if cache_path is not None:
                _atomic_torch_save(
                    {"images": self.images, "labels": self.labels, "angles": self.angles_tensor},
                    cache_path,
                )

    def __len__(self):
        return int(self.images.shape[0])

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx]), int(self.angles_tensor[idx])


class RandomC4AugmentedDataset(Dataset):
    """Returns one random C4-rotated view per access; repeats the base set to match update counts."""

    def __init__(self, base_dataset, angles=(0, 90, 180, 270), repeats=4):
        self.base_dataset = base_dataset
        self.angles = tuple(angles)
        self.repeats = int(repeats)

    def __len__(self):
        return len(self.base_dataset) * self.repeats

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx % len(self.base_dataset)]
        angle_idx = torch.randint(len(self.angles), size=(1,)).item()
        angle = self.angles[angle_idx]
        return T.functional.rotate(img, angle), label, angle


class RepeatedDataset(Dataset):
    """Repeats the base dataset N times to match iterations/epoch with the augmented sets."""

    def __init__(self, base_dataset, repeats=4):
        self.base_dataset = base_dataset
        self.repeats = repeats

    def __len__(self):
        return len(self.base_dataset) * self.repeats

    def __getitem__(self, idx):
        return self.base_dataset[idx % len(self.base_dataset)]


class EquivarianceEvalSet(Dataset):
    """Yields the full C4 orbit ``(img_0, img_90, img_180, img_270, label)`` for equivariance eval."""

    def __init__(self, base_dataset, n_samples=2000, cache_path=None):
        self.base = base_dataset
        loaded = False
        if cache_path is not None and os.path.exists(cache_path):
            payload = torch.load(cache_path, map_location="cpu", weights_only=False)
            self.img_0 = payload["img_0"]
            self.img_90 = payload["img_90"]
            self.img_180 = payload["img_180"]
            self.img_270 = payload["img_270"]
            self.labels = payload["labels"]
            loaded = True

        if not loaded:
            n_samples = min(int(n_samples), len(base_dataset))
            self.indices = torch.randperm(len(base_dataset))[:n_samples]

            img_0_list, img_90_list, img_180_list, img_270_list, labels = [], [], [], [], []
            for idx in self.indices.tolist():
                img, label = base_dataset[idx]
                img_0_list.append(img)
                img_90_list.append(T.functional.rotate(img, 90))
                img_180_list.append(T.functional.rotate(img, 180))
                img_270_list.append(T.functional.rotate(img, 270))
                labels.append(int(label))

            self.img_0 = torch.stack(img_0_list, dim=0)
            self.img_90 = torch.stack(img_90_list, dim=0)
            self.img_180 = torch.stack(img_180_list, dim=0)
            self.img_270 = torch.stack(img_270_list, dim=0)
            self.labels = torch.tensor(labels, dtype=torch.long)

            if cache_path is not None:
                _atomic_torch_save(
                    {
                        "img_0": self.img_0,
                        "img_90": self.img_90,
                        "img_180": self.img_180,
                        "img_270": self.img_270,
                        "labels": self.labels,
                    },
                    cache_path,
                )

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return (
            self.img_0[idx],
            self.img_90[idx],
            self.img_180[idx],
            self.img_270[idx],
            int(self.labels[idx]),
        )


_DATASETS = {
    "MNIST": torchvision.datasets.MNIST,
    "FashionMNIST": torchvision.datasets.FashionMNIST,
    "CIFAR10": torchvision.datasets.CIFAR10,
    "CIFAR100": torchvision.datasets.CIFAR100,
    "NCT_CRC_HE": NCTCRCHE,
}

_NUM_CLASSES = {
    "MNIST": 10,
    "FashionMNIST": 10,
    "CIFAR10": 10,
    "CIFAR100": 100,
    "NCT_CRC_HE": len(NCTCRCHE.classes),
}


def get_num_classes(dataset_name):
    if dataset_name not in _NUM_CLASSES:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return _NUM_CLASSES[dataset_name]


def get_loaders(
    dataset_name="MNIST",
    batch_size=128,
    train_size=None,
    use_augmentation=True,
    n_eval_samples=None,
    augmentation_mode=None,
    use_disk_cache=True,
    cache_dir="./data/precomputed_aug",
    data_root="./data",
    num_workers=4,
    image_size=None,
):
    """Build train / test / equivariance loaders.

    Args:
        dataset_name: One of {MNIST, FashionMNIST, CIFAR10, CIFAR100, NCT_CRC_HE}.
        batch_size: Batch size for all loaders.
        train_size: N_0, number of base training samples (None = use all).
        use_augmentation: Deprecated boolean switch, kept for compatibility.
        augmentation_mode: One of {"without", "full", "random"}; takes precedence
            over ``use_augmentation`` when provided.
        num_workers: DataLoader workers (does not affect results for the
            deterministic "without"/"full" modes).
        image_size: NCT_CRC_HE only -- patch side length (None = native 224).
            Its C4 orbit sets are always built in memory (``use_disk_cache`` is
            ignored): the processed uint8 arrays already act as the on-disk
            cache, and float32 orbit caches would take ~12 GB per N_0 = 5000
            subset at 224 px.
    """
    if augmentation_mode is None:
        augmentation_mode = "full" if use_augmentation else "without"
    if augmentation_mode not in ("without", "full", "random"):
        raise ValueError(f"Unknown augmentation_mode: {augmentation_mode}")
    if dataset_name not in _DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_kwargs = {}
    if dataset_name == "NCT_CRC_HE":
        dataset_kwargs["image_size"] = DEFAULT_IMAGE_SIZE if image_size is None else int(image_size)
        use_disk_cache = False
    elif image_size is not None:
        raise ValueError(f"image_size is only supported for NCT_CRC_HE, not {dataset_name}")

    transform = get_base_transform(dataset_name)
    dataset_cls = _DATASETS[dataset_name]
    base_train = dataset_cls(root=data_root, train=True, download=True, transform=transform, **dataset_kwargs)
    base_test = dataset_cls(root=data_root, train=False, download=True, transform=transform, **dataset_kwargs)

    if train_size is not None and train_size < len(base_train):
        indices = torch.randperm(len(base_train))[:train_size]
        base_train = torch.utils.data.Subset(base_train, indices)

    dataset_cache_dir = os.path.join(cache_dir, dataset_name) if use_disk_cache else cache_dir
    train_subset_sig = _subset_signature(base_train)
    eval_n = len(base_test) if n_eval_samples is None else min(int(n_eval_samples), len(base_test))

    if augmentation_mode == "full":
        train_cache_path = None
        if use_disk_cache:
            train_cache_path = _build_cache_path(
                dataset_cache_dir,
                prefix="train_c4_full_v1",
                dataset_name=dataset_name,
                suffix=f"{train_subset_sig}_angles-0-90-180-270",
            )
        train_dataset = C4AugmentedDataset(base_train, angles=(0, 90, 180, 270), cache_path=train_cache_path)
    elif augmentation_mode == "random":
        train_dataset = RandomC4AugmentedDataset(base_train, angles=[0, 90, 180, 270], repeats=4)
    else:  # "without": repeat 4x to keep iterations/epoch constant for a fair comparison.
        train_dataset = RepeatedDataset(base_train, repeats=4)

    test_dataset = base_test

    if n_eval_samples is None:
        n_eval_samples = len(base_test)
    eval_cache_path = None
    if use_disk_cache:
        eval_cache_path = _build_cache_path(
            dataset_cache_dir,
            prefix="equiv_eval_c4_v1",
            dataset_name=dataset_name,
            suffix=f"n-{eval_n}",
        )
    eval_dataset = EquivarianceEvalSet(base_test, n_samples=n_eval_samples, cache_path=eval_cache_path)

    worker_kwargs = {}
    if num_workers > 0:
        worker_kwargs = {"persistent_workers": True, "prefetch_factor": 4}

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=True,
        **worker_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(num_workers // 2, 0),
        pin_memory=False,
    )
    equiv_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        **worker_kwargs,
    )
    return train_loader, test_loader, equiv_loader
