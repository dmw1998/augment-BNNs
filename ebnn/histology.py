"""NCT-CRC-HE colorectal-cancer histology dataset (Kather et al., 2018).

Source: https://zenodo.org/records/1214456 (CC-BY 4.0).

* train split -- ``NCT-CRC-HE-100K``: 100,000 Macenko-normalised H&E patches from 86 slides.
* test  split -- ``CRC-VAL-HE-7K``: 7,180 patches from 50 other patients (no patient overlap).

All patches are 224x224 RGB at 0.5 um/px and carry one of nine tissue labels (ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM; indices follow the sorted folder names, as in ``ImageFolder``).  Histology has no canonical orientation, so the labels are invariant under the C4 rotations used throughout the package.

On first use the Zenodo zips are downloaded into ``<root>/NCT_CRC_HE/raw``.  Each split is then decoded once -- straight from the zip, so the 100k small TIFFs never touch the file system -- optionally resized to ``image_size``, and stored as a uint8 ``(N, H, W, 3)`` array under ``<root>/NCT_CRC_HE/processed`` that is memory-mapped at load time.  Preparation holds an exclusive lock file, so run ``scripts/prepare_nct_crc_he.py`` once before launching parallel jobs; otherwise all jobs but one fail fast instead of each downloading / writing the arrays.
"""

import io
import os
import zipfile

import numpy as np
from torch.utils.data import Dataset

ZENODO_URL = "https://zenodo.org/records/1214456/files/{name}?download=1"
NATIVE_IMAGE_SIZE = 224
DEFAULT_IMAGE_SIZE = NATIVE_IMAGE_SIZE

# train flag -> (Zenodo archive stem, md5 of the zip, number of patches)
_SPLITS = {
    True: ("NCT-CRC-HE-100K", "6fd702d11df6292bc054397ae038a464", 100000),
    False: ("CRC-VAL-HE-7K", "2fd1651b4f94ebd818ebf90ad2b6ce06", 7180),
}
_IMAGE_EXTENSIONS = (".tif", ".tiff")


class NCTCRCHE(Dataset):
    """NCT-CRC-HE-100K (``train=True``) / CRC-VAL-HE-7K (``train=False``).

    Mirrors the torchvision dataset interface used in :mod:`ebnn.data`: ``__getitem__`` returns ``(transform(PIL image), label)``.
    """

    base_folder = "NCT_CRC_HE"
    classes = ("ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM")

    def __init__(self, root, train=True, transform=None, download=False, image_size=DEFAULT_IMAGE_SIZE):
        self.root = os.path.expanduser(root)
        self.train = bool(train)
        self.transform = transform
        self.image_size = int(image_size)
        if self.image_size < 1:
            raise ValueError(f"image_size must be positive, got {image_size}")
        self.split, self.md5, self.expected_len = _SPLITS[self.train]
        self._data = None

        if not self._check_processed():
            self._prepare(download)
        self.targets = np.load(self.labels_path).tolist()
        if len(self.targets) != self.expected_len:
            print(f"Warning: {self.split} has {len(self.targets)} patches (expected {self.expected_len}).")

    # ------------------------------------------------------------------ #
    # Paths + lazy memory map
    # ------------------------------------------------------------------ #
    @property
    def raw_folder(self):
        return os.path.join(self.root, self.base_folder, "raw")

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.base_folder, "processed")

    @property
    def images_path(self):
        return os.path.join(self.processed_folder, f"{self.split}_{self.image_size}px_images.npy")

    @property
    def labels_path(self):
        return os.path.join(self.processed_folder, f"{self.split}_{self.image_size}px_labels.npy")

    @property
    def data(self):
        """uint8 ``(N, H, W, 3)`` array, memory-mapped on first access in each process."""
        if self._data is None:
            self._data = np.load(self.images_path, mmap_mode="r")
        return self._data

    def __getstate__(self):
        # Never pickle the memory map (e.g. for spawn-based DataLoader workers);
        # each process re-opens it lazily.
        state = self.__dict__.copy()
        state["_data"] = None
        return state

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.fromarray(np.array(self.data[idx]))
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]

    # ------------------------------------------------------------------ #
    # Download + one-off preprocessing
    # ------------------------------------------------------------------ #
    def _check_processed(self):
        return os.path.exists(self.images_path) and os.path.exists(self.labels_path)

    def download(self):
        """Fetch this split's zip from Zenodo (md5-checked; a no-op when the zip or extracted folder exists)."""
        if os.path.isdir(os.path.join(self.raw_folder, self.split)):
            return
        from torchvision.datasets.utils import download_url

        name = f"{self.split}.zip"
        download_url(ZENODO_URL.format(name=name), self.raw_folder, filename=name, md5=self.md5)

    def _prepare(self, download):
        os.makedirs(self.processed_folder, exist_ok=True)
        lock_path = os.path.join(self.processed_folder, f".{self.split}_{self.image_size}px.lock")
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            raise RuntimeError(
                f"{lock_path} exists: another process is preparing {self.split}. Wait for it to finish "
                "(or delete the lock if that process died); run scripts/prepare_nct_crc_he.py once "
                "before launching parallel jobs."
            ) from None
        os.close(fd)
        try:
            if not self._check_processed():  # another process may have finished meanwhile
                if download:
                    self.download()
                self._process()
        finally:
            os.remove(lock_path)

    def _label(self, member):
        cls = member.split("/")[-2]
        if cls not in self.classes:
            raise RuntimeError(
                f"Unexpected class folder {cls!r} in {member!r} (expected one of {self.classes})."
            )
        return self.classes.index(cls)

    def _process(self):
        from PIL import Image

        folder = os.path.join(self.raw_folder, self.split)
        zip_path = os.path.join(self.raw_folder, f"{self.split}.zip")
        archive = None
        if os.path.isdir(folder):  # manually extracted copy
            members = sorted(
                f"{self.split}/{cls}/{fname}"
                for cls in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, cls))
                for fname in os.listdir(os.path.join(folder, cls))
                if fname.lower().endswith(_IMAGE_EXTENSIONS)
            )

            def read(member):
                with open(os.path.join(self.raw_folder, member), "rb") as f:
                    return f.read()

        elif os.path.exists(zip_path):
            archive = zipfile.ZipFile(zip_path)
            members = sorted(n for n in archive.namelist() if n.lower().endswith(_IMAGE_EXTENSIONS))
            read = archive.read
        else:
            raise RuntimeError(
                f"{self.split} not found in {self.raw_folder}: pass download=True or run "
                "scripts/prepare_nct_crc_he.py."
            )

        n, s = len(members), self.image_size
        tmp_images = f"{self.images_path}.tmp-{os.getpid()}"
        tmp_labels = f"{self.labels_path}.tmp-{os.getpid()}"
        try:
            labels = np.asarray([self._label(m) for m in members], dtype=np.int64)
            print(f"Preparing {self.split}: decoding {n} patches to {s}x{s} uint8 -> {self.images_path}")
            images = np.lib.format.open_memmap(tmp_images, mode="w+", dtype=np.uint8, shape=(n, s, s, 3))
            for i, member in enumerate(members):
                with Image.open(io.BytesIO(read(member))) as img:
                    rgb = img.convert("RGB")
                if rgb.size != (s, s):
                    rgb = rgb.resize((s, s), Image.BILINEAR)
                images[i] = np.asarray(rgb, dtype=np.uint8)
                if (i + 1) % 10000 == 0 or i + 1 == n:
                    print(f"  {i + 1}/{n}")
            images.flush()
            del images
            with open(tmp_labels, "wb") as f:
                np.save(f, labels)
            # Labels first, images last: the dataset counts as processed only once both exist.
            os.replace(tmp_labels, self.labels_path)
            os.replace(tmp_images, self.images_path)
        except BaseException:
            for path in (tmp_images, tmp_labels):
                if os.path.exists(path):
                    os.remove(path)
            raise
        finally:
            if archive is not None:
                archive.close()
