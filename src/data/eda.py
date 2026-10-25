"""Exploratory data analysis (EDA) for CUB-200-2011 with captions (Task 1.1).

Each analysis is one "step". A step writes figures to <out>/figures, tables to
<out>/tables, and its key numbers to <out>/summary.json.

Usage
-----
    python -m src.data.eda --out outputs/eda                          # all steps
    python -m src.data.eda --out outputs/eda --steps inspect classes  # some steps
    python -m src.data.eda --out outputs/eda_test --sample 300        # quick test run

The per-image pass (used by the "images" and "quality" steps) is slow, so its
result is cached in <out>/tables/image_stats*.csv. Use --recompute to redo it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import textwrap
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageEnhance, ImageOps  # noqa: E402

TRAIN_REPO = "Multimodal-Fatima/CUB_train"
TEST_REPO = "Multimodal-Fatima/CUB_test"
STATS_SIZE = 256  # pixel statistics are computed on images resized to 256x256
# Largest per-pixel max-min channel spread (0-255) below this = no color anywhere = grayscale.
# A mean over the image fails both ways: color birds on big gray sky/water score low, and a
# black-and-white photo with one colored eye scores lower than a fully gray one
GRAY_MAX_CHROMA = 10
NEAR_DUP_THRESHOLDS_COMPARED = (2, 4, 6)  # dHash bit distances reported side by side
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STOPWORDS = {
    "a", "an", "the", "and", "with", "has", "is", "of", "this", "bird", "its", "on",
    "in", "that", "it", "are", "to", "as", "at", "for", "very", "while", "which", "but",
}
COLOR_WORDS = {
    "black", "white", "gray", "grey", "brown", "red", "orange", "yellow", "green",
    "blue", "purple", "pink", "tan", "buff", "olive", "rufous", "chestnut", "crimson",
    "scarlet", "golden", "gold", "rust", "cream", "beige", "iridescent",
}


# --------------------------------------------------------------------------
# Section 0 - helpers and shared state
# --------------------------------------------------------------------------
@dataclass
class EDAContext:
    """Everything the steps share: the two splits, output folders, and the summary."""

    train: object
    test: object
    out: Path
    seed: int = 42
    sample: int | None = None
    recompute: bool = False
    near_dup_threshold: int = 6  # 4 misses a real train/test duplicate on CUB (Marsh Wren, d=6)
    dup_review: Path | None = None  # hand-written verdicts for near-duplicate pairs
    cub_meta: Path | None = None  # folder with the official CUB text files (optional)
    summary: dict = field(default_factory=dict)
    _metadata: pd.DataFrame | None = None
    _image_df: pd.DataFrame | None = None
    _captions: tuple | None = None

    def __post_init__(self):
        self.fig_dir = self.out / "figures"
        self.table_dir = self.out / "tables"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir.mkdir(parents=True, exist_ok=True)
        summary_path = self.out / "summary.json"
        if summary_path.exists():  # keep results of steps run earlier
            self.summary = json.loads(summary_path.read_text())

    def split(self, name: str):
        return self.train if name == "train" else self.test

    def save_summary(self):
        (self.out / "summary.json").write_text(json.dumps(self.summary, indent=2, default=_to_builtin))

    @property
    def metadata(self) -> pd.DataFrame:
        if self._metadata is None:
            self._metadata = build_metadata(self.train, self.test)
        return self._metadata

    @property
    def image_df(self) -> pd.DataFrame:
        if self._image_df is None:
            self._image_df = load_or_compute_image_stats(self)
        return self._image_df

    @property
    def captions(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._captions is None:
            self._captions = build_caption_frames(self)
        return self._captions


def _to_builtin(obj):
    """Make numpy numbers JSON-serialisable."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialise {type(obj)}")


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def describe(series: pd.Series) -> dict:
    """Min / percentiles / max of a numeric column, rounded for the summary."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    q = s.quantile([0.05, 0.5, 0.95])
    return {
        "min": round(float(s.min()), 3), "p5": round(float(q[0.05]), 3),
        "median": round(float(q[0.5]), 3), "mean": round(float(s.mean()), 3),
        "p95": round(float(q[0.95]), 3), "max": round(float(s.max()), 3),
    }


def normalize_name(text: str) -> str:
    """'001.Black_footed_Albatross' / 'Black footed Albatross' -> 'black footed albatross'."""
    text = re.sub(r"^\d+\.", "", str(text)).lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def species_from_file_name(file_name: str) -> str:
    """'Black_Footed_Albatross_0007_796138.jpg' -> 'black footed albatross'."""
    stem = Path(str(file_name)).stem
    return normalize_name(stem.rsplit("_", 2)[0])


def show_image_grid(ctx: EDAContext, items, path: Path, title: str, ncols: int = 4):
    """items: list of (split, idx, caption). Draws the images in a grid with captions."""
    items = list(items)[:16]
    if not items:
        path.unlink(missing_ok=True)  # don't leave a grid from an earlier run that no longer applies
        return
    nrows = int(np.ceil(len(items) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (split, idx, caption) in zip(axes.flat, items):
        try:
            img = ctx.split(split)[int(idx)]["image"].convert("RGB")
            ax.imshow(img)
        except Exception as exc:  # corrupted file: show the error instead of crashing
            ax.text(0.5, 0.5, f"cannot decode\n{type(exc).__name__}", ha="center", va="center")
        ax.set_title(textwrap.fill(str(caption), 34), fontsize=8)
    fig.suptitle(title)
    save_fig(fig, path)


# --------------------------------------------------------------------------
# Section 1 - loading and inspection
# --------------------------------------------------------------------------
def load_splits(sample: int | None = None, seed: int = 42, cache_dir: str | None = None):
    from datasets import load_dataset

    train = load_dataset(TRAIN_REPO, split="train", cache_dir=cache_dir)
    test = load_dataset(TEST_REPO, split="test", cache_dir=cache_dir)
    if sample:  # small random subset for quick test runs
        train = train.shuffle(seed=seed).select(range(min(sample, len(train))))
        test = test.shuffle(seed=seed).select(range(min(sample, len(test))))
    return train, test


OFFICIAL_ARCHIVE_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
OFFICIAL_TEXT_FILES = ("images", "train_test_split", "image_class_labels", "classes", "bounding_boxes")


def fetch_official_metadata(meta_dir: Path):
    """Download the official CUB text files (not the images) into meta_dir.

    They sit after the images inside the 1.1 GB archive, so the whole archive is streamed
    once, but only the small text files are written to disk.
    """
    import tarfile
    import urllib.request

    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    wanted = {f"CUB_200_2011/{name}.txt" for name in OFFICIAL_TEXT_FILES}
    # data.caltech.edu answers 403 to Python's default User-Agent
    request = urllib.request.Request(OFFICIAL_ARCHIVE_URL, headers={"User-Agent": "curl/8.0"})
    print(f"  streaming {OFFICIAL_ARCHIVE_URL} (about 1.1 GB, a few minutes)")
    with urllib.request.urlopen(request) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
        for member in tar:
            if member.name in wanted and member.isfile():
                (meta_dir / Path(member.name).name).write_bytes(tar.extractfile(member).read())
                wanted.discard(member.name)
                print(f"  saved {meta_dir / Path(member.name).name}")
                if not wanted:
                    break
    if wanted:
        raise RuntimeError(f"files not found in the archive: {sorted(wanted)}")


def load_official_metadata(meta_dir: Path | None) -> pd.DataFrame | None:
    """Official CUB split, class and bird bounding box per file name, or None if not downloaded.

    The Hugging Face copy has no boxes, so these text files come from the original
    CUB_200_2011.tgz (images.txt, train_test_split.txt, image_class_labels.txt, classes.txt,
    bounding_boxes.txt). Boxes are x, y, width, height in original pixels.
    """
    if meta_dir is None or not all((Path(meta_dir) / f"{n}.txt").exists() for n in OFFICIAL_TEXT_FILES):
        return None

    def read(name, cols):
        return pd.read_csv(Path(meta_dir) / f"{name}.txt", sep=" ", header=None, names=cols)

    df = (read("images", ["image_id", "path"])
          .merge(read("train_test_split", ["image_id", "is_train"]), on="image_id")
          .merge(read("image_class_labels", ["image_id", "class_id"]), on="image_id")
          .merge(read("classes", ["class_id", "class_name"]), on="class_id")
          .merge(read("bounding_boxes", ["image_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]), on="image_id"))
    return pd.DataFrame({
        "file_name": df.path.map(lambda p: Path(p).name),
        "official_split": np.where(df.is_train == 1, "train", "test"),
        "official_class": df.class_name.map(normalize_name),
        "bbox_x": df.bbox_x, "bbox_y": df.bbox_y, "bbox_w": df.bbox_w, "bbox_h": df.bbox_h,
    })


def label_names(ds) -> list[str] | None:
    feature = ds.features.get("label")
    return list(feature.names) if hasattr(feature, "names") else None


def build_metadata(train, test) -> pd.DataFrame:
    """One row per image with everything except the pixels (fast: no image decoding)."""
    frames = []
    for split, ds in (("train", train), ("test", test)):
        cols = [c for c in ("id", "label", "file_name", "description") if c in ds.column_names]
        df = ds.select_columns(cols).to_pandas()
        df.insert(0, "idx", range(len(df)))
        df.insert(0, "split", split)
        names = label_names(ds)
        if names is not None:
            df["label_name"] = [names[int(v)] if pd.notna(v) else None for v in df["label"]]
        else:
            df["label_name"] = df["label"].astype(str)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def step_inspect(ctx: EDAContext):
    """Schema, row counts, id uniqueness, label / file-name consistency, sample grid."""
    meta = ctx.metadata
    info = {}
    for split in ("train", "test"):
        ds = ctx.split(split)
        info[split] = {
            "num_rows": len(ds),
            "columns": {name: str(feat) for name, feat in ds.features.items()},
        }
        print(f"  {split}: {len(ds)} rows, columns = {ds.column_names}")

    if "id" in meta:
        info["duplicate_ids_within_split"] = int(meta.duplicated(["split", "id"]).sum())
        info["ids_shared_by_train_and_test"] = int(
            len(set(meta.loc[meta.split == "train", "id"]) & set(meta.loc[meta.split == "test", "id"]))
        )
        # If id is just the row number, shared ids say nothing about leakage; file_name is the real key
        info["id_is_row_index"] = {split: bool((g["id"] == g["idx"]).all()) for split, g in meta.groupby("split")}

    # Does the species in the file name agree with the label?
    if "file_name" in meta:
        meta = meta.assign(
            file_species=meta["file_name"].map(species_from_file_name),
            label_norm=meta["label_name"].map(normalize_name),
        )
        mismatch = meta[meta.file_species != meta.label_norm]
        mismatch[["split", "idx", "file_name", "label_name"]].to_csv(
            ctx.table_dir / "label_filename_mismatch.csv", index=False
        )
        info["label_filename_mismatches"] = len(mismatch)
        info["file_names_shared_by_train_and_test"] = int(
            len(set(meta.loc[meta.split == "train", "file_name"])
                & set(meta.loc[meta.split == "test", "file_name"]))
        )

    # Same split and labels as the official release? Needed for a fair benchmark against papers
    official = load_official_metadata(ctx.cub_meta)
    if official is not None and "file_name" in meta:
        joined = meta.merge(official, on="file_name", how="left")
        known = joined.official_split.notna()
        info["official_metadata"] = {
            "files_not_in_official_release": int((~known).sum()),
            "split_differs_from_official": int((known & (joined.split != joined.official_split)).sum()),
            "label_differs_from_official": int((known & (joined.label_norm != joined.official_class)).sum()),
        }

    ctx.summary["inspect"] = info

    rng = random.Random(ctx.seed)
    train_meta = ctx.metadata[ctx.metadata.split == "train"]
    picks = rng.sample(range(len(train_meta)), k=min(12, len(train_meta)))
    items = []
    for i in picks:
        row = train_meta.iloc[i]
        first_caption = str(row.get("description", "")).strip().split("\n")[0]
        items.append(("train", row.idx, f"{row.label_name}\n\"{first_caption}\""))
    show_image_grid(ctx, items, ctx.fig_dir / "01_sample_grid.png", "Random training samples")


# --------------------------------------------------------------------------
# Section 2 - class distribution
# --------------------------------------------------------------------------
def step_classes(ctx: EDAContext):
    meta = ctx.metadata
    counts = meta.groupby(["label_name", "split"]).size().unstack(fill_value=0)
    for split in ("train", "test"):
        if split not in counts:
            counts[split] = 0
    counts = counts[["train", "test"]].sort_values("train")
    counts.to_csv(ctx.table_dir / "class_counts.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, split in zip(axes, ("train", "test")):
        values = counts[split].sort_values().to_numpy()
        ax.bar(range(len(values)), values, width=1.0)
        ax.axhline(values.mean(), color="red", linestyle="--", label=f"mean = {values.mean():.1f}")
        ax.set_title(f"Images per class ({split}, sorted)")
        ax.set_xlabel("class (sorted by count)")
        ax.set_ylabel("images")
        ax.legend()
    save_fig(fig, ctx.fig_dir / "02_class_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    # Ranked by train + test: train is nearly flat, so the classes that are really small show up in test
    smallest = counts.loc[(counts["train"] + counts["test"]).nsmallest(15).index]
    ax.barh(smallest.index, smallest["train"], label="train")
    ax.barh(smallest.index, smallest["test"], left=smallest["train"], label="test", alpha=0.6)
    ax.set_title("15 smallest classes (train + test images)")
    ax.set_xlabel("images")
    ax.legend()
    save_fig(fig, ctx.fig_dir / "02_smallest_classes.png")

    info = {"num_classes": int(len(counts))}
    for split in ("train", "test"):
        c = counts[split]
        info[split] = {
            "min": int(c.min()), "max": int(c.max()), "mean": round(float(c.mean()), 2),
            "std": round(float(c.std()), 2),
            "imbalance_ratio_max_over_min": round(float(c.max() / max(c.min(), 1)), 2),
            "classes_with_zero_images": counts.index[c == 0].tolist(),
        }
    # How many validation images per class would a stratified split leave?
    min_train = int(counts["train"].min())
    info["val_images_per_class_for_smallest_class"] = {
        str(frac): int(np.floor(min_train * frac)) for frac in (0.1, 0.15, 0.2)
    }
    ctx.summary["classes"] = info


# --------------------------------------------------------------------------
# Section 3 - per-image statistics (one pass over every image) and plots
# --------------------------------------------------------------------------
def iter_raw_images(ds):
    """Yield (index, record, raw_bytes) without letting the datasets library decode.

    Decoding ourselves means one corrupted file is recorded instead of crashing
    the loop, and the raw bytes can be hashed to find exact duplicates.
    """
    from datasets import Image as HFImage

    raw = ds.cast_column("image", HFImage(decode=False))
    for i, record in enumerate(raw):
        info = record["image"] or {}
        data = info.get("bytes")
        if data is None and info.get("path"):
            data = Path(info["path"]).read_bytes()
        yield i, record, data


def dhash(img: Image.Image) -> str:
    """64-bit difference hash: similar images get hashes that differ in few bits."""
    small = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return np.packbits(bits).tobytes().hex()


def pixel_stats(img: Image.Image) -> dict:
    rgb = img.convert("RGB").resize((STATS_SIZE, STATS_SIZE), Image.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b

    g255 = gray * 255.0  # Laplacian variance: low = few sharp edges = blurry
    lap = (-4 * g255[1:-1, 1:-1] + g255[:-2, 1:-1] + g255[2:, 1:-1]
           + g255[1:-1, :-2] + g255[1:-1, 2:])

    R, G, B = r * 255.0, g * 255.0, b * 255.0  # colorfulness (Hasler & Suesstrunk, 2003)
    rg, yb = R - G, 0.5 * (R + G) - B
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    return {
        "mean_r": float(r.mean()), "mean_g": float(g.mean()), "mean_b": float(b.mean()),
        # mean of squares lets us compute the exact dataset std later
        "msq_r": float((r ** 2).mean()), "msq_g": float((g ** 2).mean()), "msq_b": float((b ** 2).mean()),
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "blur_lap_var": float(lap.var()),
        "colorfulness": float(colorfulness),
        "max_chroma": float((arr.max(axis=2) - arr.min(axis=2)).max() * 255.0),
        "dhash": dhash(img),
    }


def compute_image_stats(ds, split: str) -> pd.DataFrame:
    names = label_names(ds)
    rows = []
    for i, record, data in iter_raw_images(ds):
        label = record.get("label")
        row = {
            "split": split, "idx": i, "file_name": record.get("file_name"),
            "label_name": names[label] if names is not None and label is not None else label,
            "error": "",
        }
        try:
            if data is None:
                raise ValueError("no image bytes")
            row["md5"] = hashlib.md5(data).hexdigest()
            img = Image.open(io.BytesIO(data))
            row.update(format=img.format, mode=img.mode, width=img.size[0], height=img.size[1])
            img.load()  # forces a full decode; truncated files fail here
            row.update(pixel_stats(img))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        if (i + 1) % 1000 == 0:
            print(f"  {split}: {i + 1}/{len(ds)} images processed")
    return pd.DataFrame(rows)


def load_or_compute_image_stats(ctx: EDAContext) -> pd.DataFrame:
    suffix = f"_sample{ctx.sample}" if ctx.sample else ""
    cache = ctx.table_dir / f"image_stats{suffix}.csv"
    if cache.exists() and not ctx.recompute:
        print(f"  using cached {cache}")
        df = pd.read_csv(cache, dtype={"md5": str, "dhash": str, "error": str})
    else:
        df = pd.concat([compute_image_stats(ctx.train, "train"),
                        compute_image_stats(ctx.test, "test")], ignore_index=True)
        df.to_csv(cache, index=False)
    df["error"] = df["error"].fillna("")
    df["aspect_ratio"] = df["width"] / df["height"]
    df["shorter_side"] = df[["width", "height"]].min(axis=1)
    df["is_grayscale"] = df["mode"].isin(["L", "LA", "1", "I", "I;16", "F"]) | (
        df["max_chroma"] < GRAY_MAX_CHROMA
    )
    return df


def step_images(ctx: EDAContext):
    df = ctx.image_df
    ok = df[df.error == ""]

    # Sizes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for split, part in ok.groupby("split"):
        axes[0].scatter(part.width, part.height, s=4, alpha=0.4, label=split)
        axes[1].hist(part.aspect_ratio, bins=50, alpha=0.6, label=split)
        axes[2].hist(part.shorter_side, bins=50, alpha=0.6, label=split)
    axes[0].set(title="Width vs height", xlabel="width (px)", ylabel="height (px)")
    axes[1].set(title="Aspect ratio (width / height)", xlabel="ratio", ylabel="images")
    axes[2].axvline(224, color="red", linestyle="--", label="224 px")
    axes[2].set(title="Shorter side", xlabel="pixels", ylabel="images")
    for ax in axes:
        ax.legend()
    save_fig(fig, ctx.fig_dir / "03_image_sizes.png")

    # Brightness, contrast, sharpness, colorfulness, channel means
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("brightness", "Brightness (0-1)", False), ("contrast", "Contrast (gray std)", False),
        ("blur_lap_var", "Sharpness: Laplacian variance (log scale)", True),
        ("colorfulness", "Colorfulness (0 = gray)", False),
    ]
    for ax, (col, title, log) in zip(axes.flat, panels):
        for split, part in ok.groupby("split"):
            values = np.log10(part[col] + 1) if log else part[col]
            ax.hist(values, bins=50, alpha=0.6, label=split)
        ax.set(title=title, ylabel="images")
        ax.legend()
    train_ok = ok[ok.split == "train"]
    for col, color in (("mean_r", "red"), ("mean_g", "green"), ("mean_b", "blue")):
        axes.flat[4].hist(train_ok[col], bins=50, alpha=0.5, color=color, label=col)
    axes.flat[4].set(title="Per-image channel means (train)", ylabel="images")
    axes.flat[4].legend()
    modes = df.groupby(["split", "mode"]).size().unstack(fill_value=0)
    modes.T.plot.bar(ax=axes.flat[5])
    axes.flat[5].set(title="Image modes", ylabel="images")
    save_fig(fig, ctx.fig_dir / "03_image_quality.png")

    # Example grids so every automatic flag is checked by eye
    def grid(part, name, title, caption_col):
        items = [(r.split, r.idx, f"{r.label_name}\n{caption_col}={getattr(r, caption_col):.2f}")
                 for r in part.itertuples()]
        show_image_grid(ctx, items, ctx.fig_dir / f"03_examples_{name}.png", title)

    grid(ok.nsmallest(12, "colorfulness"), "least_colorful", "Least colorful images", "colorfulness")
    grid(ok.nsmallest(12, "blur_lap_var"), "blurriest", "Blurriest images", "blur_lap_var")
    grid(ok.nsmallest(8, "brightness"), "darkest", "Darkest images", "brightness")
    grid(ok.nsmallest(8, "shorter_side"), "smallest", "Smallest images", "shorter_side")
    # Borderline cases first: they are the likely false positives (true L-mode images score 0)
    grid(ok[ok.is_grayscale].sort_values("max_chroma", ascending=False), "grayscale",
         "Images flagged as grayscale", "max_chroma")

    gray = df[df.is_grayscale]
    gray[["split", "idx", "file_name", "label_name", "mode", "max_chroma"]].to_csv(
        ctx.table_dir / "grayscale_images.csv", index=False
    )
    not_rgb = df[df["mode"].notna() & (df["mode"] != "RGB")]
    not_rgb[["split", "idx", "file_name", "mode"]].to_csv(ctx.table_dir / "non_rgb_images.csv", index=False)

    # Normalisation statistics from the TRAINING split only (test must stay unseen)
    mean = [train_ok[f"mean_{c}"].mean() for c in "rgb"]
    std = [float(np.sqrt(train_ok[f"msq_{c}"].mean() - m ** 2)) for c, m in zip("rgb", mean)]

    ctx.summary["images"] = {
        "n_images": int(len(df)),
        "n_decode_errors": int((df.error != "").sum()),
        "modes": {split: part["mode"].value_counts().to_dict() for split, part in df.groupby("split")},
        "formats": df["format"].value_counts().to_dict(),
        "width": describe(ok.width), "height": describe(ok.height),
        "shorter_side": describe(ok.shorter_side), "aspect_ratio": describe(ok.aspect_ratio),
        "n_shorter_side_below_224": int((ok.shorter_side < 224).sum()),
        "n_grayscale": {split: int(part.is_grayscale.sum()) for split, part in df.groupby("split")},
        "brightness": describe(ok.brightness), "contrast": describe(ok.contrast),
        "blur_lap_var": describe(ok.blur_lap_var), "colorfulness": describe(ok.colorfulness),
        "normalization_train": {"mean": [round(m, 4) for m in mean], "std": [round(s, 4) for s in std]},
        "normalization_imagenet": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
    }


# --------------------------------------------------------------------------
# Section 4 - data quality: corrupted files, missing values, duplicates
# --------------------------------------------------------------------------
def near_duplicate_pairs(hashes: list[str], threshold: int, chunk: int = 1024):
    """All pairs (i, j, distance) whose dHashes differ in <= threshold bits."""
    bits = np.array([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8)) for h in hashes],
                    dtype=np.float32)
    ones = bits.sum(axis=1)
    pairs = []
    for start in range(0, len(bits), chunk):
        block = bits[start:start + chunk]
        # Hamming distance = |a| + |b| - 2 * (a . b) for 0/1 vectors
        dist = ones[start:start + chunk, None] + ones[None, :] - 2 * (block @ bits.T)
        rows, cols = np.nonzero(dist <= threshold)
        for r, c in zip(rows, cols):
            i = r + start
            if i < c:
                pairs.append((i, int(c), int(round(dist[r, c]))))
    return pairs


def attach_review(near: pd.DataFrame, review_path: Path | None) -> pd.DataFrame:
    """Add the by-eye verdict of each pair from a CSV with columns file_a, file_b, verdict, note.

    Hashes give false positives, so a pair only counts as a duplicate once someone has looked at it.
    Pairs are matched by file name in either order; pairs missing from the file are "unreviewed".
    """
    verdicts = {}
    if review_path is not None and Path(review_path).exists():
        for r in pd.read_csv(review_path).itertuples():
            verdicts[frozenset((r.file_a, r.file_b))] = (r.verdict, r.note)
    found = [verdicts.get(frozenset(pair), ("unreviewed", "")) for pair in zip(near.file_a, near.file_b)]
    return near.assign(verdict=[v for v, _ in found], note=[n for _, n in found])


def step_quality(ctx: EDAContext):
    df, meta = ctx.image_df, ctx.metadata
    info = {}

    errors = df[df.error != ""]
    errors[["split", "idx", "file_name", "error"]].to_csv(ctx.table_dir / "decode_errors.csv", index=False)
    info["n_decode_errors"] = int(len(errors))

    missing = {}
    for col in ("label", "file_name", "description"):
        if col in meta:
            empty = meta[col].isna() | (meta[col].astype(str).str.strip() == "")
            missing[col] = int(empty.sum())
    info["missing_values"] = missing

    # Exact duplicates: identical file bytes
    hashed = df[df.md5.notna()]
    groups = hashed.groupby("md5").filter(lambda g: len(g) > 1).sort_values("md5")
    groups[["md5", "split", "idx", "file_name", "label_name"]].to_csv(
        ctx.table_dir / "exact_duplicates.csv", index=False
    )
    splits_per_group = groups.groupby("md5").split.nunique()
    info["exact_duplicate_groups"] = int(groups.md5.nunique())
    info["exact_duplicate_groups_across_train_test"] = int((splits_per_group > 1).sum())

    # Near duplicates: same picture after resizing / recompression / small edits.
    # Search once at the largest threshold so the compared thresholds come from a single run.
    valid = df[df.dhash.notna()].reset_index(drop=True)
    search = max(ctx.near_dup_threshold, *NEAR_DUP_THRESHOLDS_COMPARED)
    pairs = near_duplicate_pairs(valid.dhash.tolist(), search)
    rows = []
    for i, j, d in pairs:
        a, b = valid.iloc[i], valid.iloc[j]
        if a.md5 == b.md5:
            continue  # already counted as an exact duplicate
        rows.append({
            "split_a": a.split, "idx_a": a.idx, "file_a": a.file_name, "label_a": a.label_name,
            "split_b": b.split, "idx_b": b.idx, "file_b": b.file_name, "label_b": b.label_name,
            "hamming": d, "same_label": a.label_name == b.label_name,
        })
    near = pd.DataFrame(rows, columns=["split_a", "idx_a", "file_a", "label_a", "split_b", "idx_b",
                                       "file_b", "label_b", "hamming", "same_label"])
    info["near_duplicate_pairs_by_threshold"] = {
        t: int((near.hamming <= t).sum()) for t in NEAR_DUP_THRESHOLDS_COMPARED
    }
    near = attach_review(near[near.hamming <= ctx.near_dup_threshold], ctx.dup_review)
    near = near.assign(cross=near.split_a != near.split_b)
    near.sort_values("hamming").to_csv(ctx.table_dir / "near_duplicates.csv", index=False)
    info["near_duplicate_threshold_bits"] = ctx.near_dup_threshold
    info["near_duplicate_pairs"] = int(len(near))
    info["near_duplicate_pairs_across_train_test"] = int(near.cross.sum())
    info["near_duplicate_pairs_with_different_labels"] = int((~near.same_label).sum()) if len(near) else 0
    info["near_duplicate_verdicts"] = near.verdict.value_counts().to_dict()

    # Leakage: train images whose photo is also in test. Drop the train copy, never the test one,
    # so test results stay comparable to published ones
    leak = [(r.idx, r.file_name, "exact copy in test")
            for r in groups[groups.md5.isin(splits_per_group[splits_per_group > 1].index)
                            & (groups.split == "train")].itertuples()]
    for r in near[near.cross & (near.verdict == "duplicate")].itertuples():
        train_side, test_side = ((r.idx_a, r.file_a), (r.idx_b, r.file_b)) if r.split_a == "train" \
            else ((r.idx_b, r.file_b), (r.idx_a, r.file_a))
        leak.append((*train_side, f"same photo as test #{test_side[0]} {test_side[1]}"))
    pd.DataFrame(leak, columns=["train_idx", "train_file", "reason"]).to_csv(
        ctx.table_dir / "leakage_train_to_drop.csv", index=False
    )
    info["train_images_to_drop_for_leakage"] = len(leak)

    # Every pair side by side with its verdict: confirmed duplicates first (train/test leaks on top)
    fig_path = ctx.fig_dir / "04_near_duplicates.png"
    fig_path.unlink(missing_ok=True)
    if len(near):
        rank = near.verdict.map({"duplicate": 0, "unreviewed": 1}).fillna(2)
        top = (near.assign(rank=rank).sort_values(["rank", "cross", "hamming"], ascending=[True, False, True])
               .head(12))
        verdict_text = {"duplicate": ("DUPLICATE", "tab:green"), "unreviewed": ("not reviewed", "gray")}
        fig, axes = plt.subplots(len(top), 2, figsize=(7, 3.2 * len(top)), squeeze=False)
        for row_axes, r in zip(axes, top.itertuples()):
            text, color = verdict_text.get(r.verdict, ("false positive: different photos", "tab:red"))
            if r.verdict == "duplicate" and r.cross:
                text += " - train/test leak"
            for ax, split, idx, label in ((row_axes[0], r.split_a, r.idx_a, r.label_a),
                                          (row_axes[1], r.split_b, r.idx_b, r.label_b)):
                ax.imshow(ctx.split(split)[int(idx)]["image"].convert("RGB"))
                ax.set_title(f"{split} #{idx}  {label}  (d={r.hamming})\n{text}", fontsize=8, color=color)
                ax.axis("off")
        fig.suptitle("Near-duplicate candidates and their by-eye verdicts", y=1.01)
        save_fig(fig, fig_path)

    ctx.summary["quality"] = info


# --------------------------------------------------------------------------
# Section 5 - captions
# --------------------------------------------------------------------------
def clean_caption(text: str) -> str:
    """Proposed cleaning rule (Member 2 implements the final version in the pipeline)."""
    text = text.lower()
    text = re.sub(r"[’']", "", text)  # bird's -> birds
    text = re.sub(r"[^a-z0-9 ]+", " ", text)  # punctuation and other symbols -> space
    return re.sub(r"\s+", " ", text).strip()


def edits1(word: str) -> set[str]:
    """Every string one delete / swap / replace / insert away from word (Norvig's speller)."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    swaps = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in letters]
    inserts = [a + c + b for a, b in splits for c in letters]
    return set(deletes + swaps + replaces + inserts)


def build_caption_frames(ctx: EDAContext):
    """Returns (one row per caption, one row per image)."""
    caption_rows, image_rows = [], []
    for r in ctx.metadata.itertuples():
        desc = r.description if isinstance(r.description, str) else ""
        lines = desc.split("\n")
        captions = [line.strip() for line in lines if line.strip()]
        image_rows.append({"split": r.split, "idx": r.idx, "n_captions": len(captions),
                           "n_empty_lines": len(lines) - len(captions)})
        for j, raw in enumerate(captions):
            cleaned = clean_caption(raw)
            caption_rows.append({
                "split": r.split, "idx": r.idx, "label_name": r.label_name, "caption_no": j,
                "raw": raw, "clean": cleaned,
                "n_words_raw": len(raw.split()), "n_tokens": len(cleaned.split()),
            })
    return pd.DataFrame(caption_rows), pd.DataFrame(image_rows)


def step_captions(ctx: EDAContext):
    caps, per_image = ctx.captions
    train_caps = caps[caps.split == "train"]
    info = {}

    # 1) Captions per image
    info["captions_per_image"] = per_image.n_captions.value_counts().sort_index().to_dict()
    info["images_without_10_captions"] = int((per_image.n_captions != 10).sum())
    info["empty_lines_total"] = int(per_image.n_empty_lines.sum())
    per_image[per_image.n_captions != 10].to_csv(ctx.table_dir / "images_without_10_captions.csv", index=False)

    # 2) Lengths -> decides the decoder's max_len
    lengths = train_caps.n_tokens
    pct = {f"p{p}": int(np.percentile(lengths, p)) for p in (50, 90, 95, 99)}
    info["tokens_per_caption_train"] = {"min": int(lengths.min()), **pct, "max": int(lengths.max())}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=range(0, int(lengths.max()) + 2), alpha=0.8)
    ax.axvline(pct["p95"], color="red", linestyle="--", label=f"95th percentile = {pct['p95']}")
    ax.set(title="Caption length (train, tokens after cleaning)", xlabel="tokens", ylabel="captions")
    ax.legend()
    save_fig(fig, ctx.fig_dir / "05_caption_lengths.png")
    caps[caps.n_tokens <= 3].to_csv(ctx.table_dir / "very_short_captions.csv", index=False)
    caps[caps.n_tokens > pct["p99"]].to_csv(ctx.table_dir / "very_long_captions.csv", index=False)
    info["very_short_captions_le3_tokens"] = int((caps.n_tokens <= 3).sum())

    # 3) Vocabulary, built from TRAIN only. min_freq is chosen without touching test: words are
    # counted on 90% of training images and <unk> is measured on the other 10%, split by image
    # because the 10 captions of one image are near-paraphrases
    train_counts = Counter(tok for c in train_caps.clean for tok in c.split())
    images = sorted(train_caps.idx.unique())
    held_out = set(random.Random(ctx.seed).sample(images, k=max(1, len(images) // 10)))
    is_held = train_caps.idx.isin(held_out)
    fit_counts = Counter(tok for c in train_caps.clean[~is_held] for tok in c.split())
    held_tokens = [tok for c in train_caps.clean[is_held] for tok in c.split()]
    total_train = sum(train_counts.values())
    vocab_rows = []
    for k in (1, 2, 3, 5, 10, 20):
        kept = {w for w, n in train_counts.items() if n >= k}
        fit_kept = {w for w, n in fit_counts.items() if n >= k}
        vocab_rows.append({
            "min_freq": k, "vocab_size": len(kept),
            "train_token_coverage_%": round(100 * sum(train_counts[w] for w in kept) / total_train, 3),
            "heldout_unk_token_rate_%": round(
                100 * sum(t not in fit_kept for t in held_tokens) / max(len(held_tokens), 1), 3),
        })
    vocab_df = pd.DataFrame(vocab_rows)
    vocab_df.to_csv(ctx.table_dir / "vocab_by_min_freq.csv", index=False)
    info["vocab_by_min_freq"] = vocab_rows
    pd.DataFrame(train_counts.most_common(), columns=["word", "count"]).to_csv(
        ctx.table_dir / "word_frequencies_train.csv", index=False
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(vocab_df.min_freq, vocab_df.vocab_size, marker="o")
    axes[0].set(title="Vocabulary size vs. min frequency", xlabel="min_freq", ylabel="words kept")
    axes[1].plot(vocab_df.min_freq, vocab_df["heldout_unk_token_rate_%"], marker="o", color="orange")
    axes[1].set(title="Held-out train tokens that become <unk>", xlabel="min_freq", ylabel="% of tokens")
    save_fig(fig, ctx.fig_dir / "05_vocab_threshold.png")

    top = [(w, n) for w, n in train_counts.most_common() if w not in STOPWORDS][:30]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh([w for w, _ in top][::-1], [n for _, n in top][::-1])
    ax.set(title="30 most frequent words (train, stopwords removed)", xlabel="count")
    save_fig(fig, ctx.fig_dir / "05_top_words.png")

    # 4) Typo candidates: rare words one edit away from a frequent word ("brwon" -> "brown")
    frequent = {w for w, n in train_counts.items() if n >= 50}
    rare = [w for w, n in train_counts.items() if n <= 2 and len(w) >= 3]
    typos = []
    for w in rare:
        matches = edits1(w) & frequent
        if matches:
            best = max(matches, key=train_counts.get)
            typos.append({"rare_word": w, "count": train_counts[w], "probably_meant": best,
                          "count_of_match": train_counts[best]})
    pd.DataFrame(typos).to_csv(ctx.table_dir / "typo_candidates.csv", index=False)
    info["words_seen_once_train"] = int(sum(1 for n in train_counts.values() if n == 1))
    info["typo_candidates"] = len(typos)

    # 5) Duplicates and odd characters
    info["duplicate_captions_within_same_image"] = int(caps.duplicated(["split", "idx", "clean"]).sum())
    repeated = train_caps.clean.value_counts()
    repeated[repeated > 1].head(50).to_csv(ctx.table_dir / "most_repeated_captions_train.csv")
    info["distinct_train_captions_used_more_than_once"] = int((repeated > 1).sum())
    non_ascii = caps[caps.raw.map(lambda s: any(ord(ch) > 127 for ch in s))]
    non_ascii.to_csv(ctx.table_dir / "non_ascii_captions.csv", index=False)
    info["captions_with_non_ascii"] = int(len(non_ascii))
    removed_chars = Counter(ch for s in caps.raw for ch in s.lower() if not re.match(r"[a-z0-9 ]", ch))
    info["characters_removed_by_cleaning"] = dict(removed_chars.most_common(20))

    # 6) Evidence for augmentation choices (train only, like every other training decision)
    tokens = train_caps.clean.str.split()
    info["%_captions_mentioning_left_or_right"] = round(
        100 * tokens.map(lambda t: "left" in t or "right" in t).mean(), 3)
    info["%_captions_with_a_color_word"] = round(
        100 * tokens.map(lambda t: bool(COLOR_WORDS & set(t))).mean(), 2)
    color_counts = Counter(t for toks in tokens for t in toks if t in COLOR_WORDS)
    fig, ax = plt.subplots(figsize=(8, 4))
    words, counts = zip(*color_counts.most_common(15)) if color_counts else ((), ())
    ax.bar(words, counts)
    ax.set(title="Color words in captions (train)", ylabel="occurrences")
    ax.tick_params(axis="x", rotation=45)
    save_fig(fig, ctx.fig_dir / "05_color_words.png")

    ctx.summary["captions"] = info


# --------------------------------------------------------------------------
# Section 6 - augmentation proposal (visual check of candidate transforms)
# These PIL versions mimic torchvision's RandomHorizontalFlip, RandomRotation,
# ColorJitter and RandomResizedCrop so the effect can be judged before
# Member 2 writes the real transforms.
# --------------------------------------------------------------------------
CROP_SCALE_LOWER_BOUNDS = (0.08, 0.25, 0.4, 0.6, 0.8)  # 0.08 is torchvision's default


def aug_rotate(img, rng, degrees=15):
    return img.rotate(rng.uniform(-degrees, degrees), resample=Image.BILINEAR)


def aug_color_jitter(img, rng, strength=0.2):
    for enhancer in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        img = enhancer(img).enhance(rng.uniform(1 - strength, 1 + strength))
    return img


def aug_hue_shift(img, rng, max_shift=0.4):
    hsv = np.asarray(img.convert("HSV")).copy()
    shift = int(rng.uniform(-max_shift, max_shift) * 255)
    hsv[..., 0] = (hsv[..., 0].astype(int) + shift) % 256
    return Image.fromarray(hsv, mode="HSV").convert("RGB")


def sample_resized_crop_box(width, height, rng, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
    """(left, top, w, h) of a RandomResizedCrop window, sampled the same way torchvision does."""
    area = width * height
    for _ in range(10):
        target = area * rng.uniform(*scale)
        aspect = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        w, h = int(round(np.sqrt(target * aspect))), int(round(np.sqrt(target / aspect)))
        if 0 < w <= width and 0 < h <= height:
            return rng.randint(0, width - w), rng.randint(0, height - h), w, h
    # Fallback: central crop with the aspect ratio clamped to the allowed range
    in_ratio = width / height
    w, h = width, height
    if in_ratio < min(ratio):
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        w = int(round(h * max(ratio)))
    return (width - w) // 2, (height - h) // 2, w, h


def aug_random_resized_crop(img, rng, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), size=224):
    left, top, w, h = sample_resized_crop_box(*img.size, rng, scale, ratio)
    return img.crop((left, top, left + w, top + h)).resize((size, size), Image.BILINEAR)


def visible_fraction(box, crop) -> float:
    """Share of the bird box (x, y, w, h) that lies inside the crop window (x, y, w, h)."""
    bx, by, bw, bh = box
    cx, cy, cw, ch = crop
    overlap_w = max(0.0, min(bx + bw, cx + cw) - max(bx, cx))
    overlap_h = max(0.0, min(by + bh, cy + ch) - max(by, cy))
    return overlap_w * overlap_h / (bw * bh) if bw * bh > 0 else 0.0


def bird_crop_analysis(ctx: EDAContext, crops_per_image: int = 10) -> dict | None:
    """How big the bird is in the frame, and how much of it each cropping policy keeps (train only)."""
    official = load_official_metadata(ctx.cub_meta)
    fig_path = ctx.fig_dir / "06_bird_size_and_crops.png"
    if official is None:
        print("  no official CUB metadata: skipping the bounding-box analysis")
        fig_path.unlink(missing_ok=True)  # don't leave results from an earlier run
        return None
    df = ctx.image_df
    train = df[(df.split == "train") & (df.error == "")].merge(official, on="file_name")
    boxes = list(zip(train.bbox_x, train.bbox_y, train.bbox_w, train.bbox_h))
    sizes = list(zip(train.width, train.height))
    bird_fraction = train.bbox_w * train.bbox_h / (train.width * train.height)

    # Validation/test policy: resize the shorter side to `resize`, then take the central 224x224
    center = {}
    for resize in (256, 224):
        kept = []
        for (w, h), box in zip(sizes, boxes):
            side = 224 * min(w, h) / resize
            kept.append(visible_fraction(box, ((w - side) / 2, (h - side) / 2, side, side)))
        kept = np.array(kept)
        center[f"resize_{resize}_center_crop_224"] = {
            "mean_bird_visible_%": round(100 * float(kept.mean()), 1),
            "%_images_bird_fully_inside": round(100 * float((kept >= 0.999).mean()), 2),
            "%_images_less_than_half_the_bird": round(100 * float((kept < 0.5).mean()), 2),
        }

    rng = random.Random(ctx.seed)
    by_scale = []
    for low in CROP_SCALE_LOWER_BOUNDS:
        kept = np.array([visible_fraction(box, sample_resized_crop_box(w, h, rng, scale=(low, 1.0)))
                         for (w, h), box in zip(sizes, boxes) for _ in range(crops_per_image)])
        by_scale.append({"scale_min": low, "mean_bird_visible_%": round(100 * kept.mean(), 1),
                         "%_crops_with_less_than_half_the_bird": round(100 * (kept < 0.5).mean(), 2)})

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(100 * bird_fraction, bins=50)
    axes[0].set(title="Bird bounding box as % of the image (train)", xlabel="% of image area",
                ylabel="images")
    scale_df = pd.DataFrame(by_scale)
    axes[1].plot(scale_df.scale_min, scale_df["%_crops_with_less_than_half_the_bird"], marker="o",
                 color="tab:red")
    axes[1].set(title="RandomResizedCrop: crops that keep less than half the bird",
                xlabel="lower bound of the crop scale", ylabel="% of random crops")
    save_fig(fig, fig_path)

    return {
        "n_train_images_with_box": int(len(train)),
        "bird_area_%_of_image": describe(100 * bird_fraction),
        "%_images_bird_under_10%_of_image": round(100 * float((bird_fraction < 0.1).mean()), 2),
        "center_crop": center,
        "random_resized_crop_by_scale_min": by_scale,
    }


def step_augment(ctx: EDAContext):
    rng = random.Random(ctx.seed)
    picks = rng.sample(range(len(ctx.train)), k=min(3, len(ctx.train)))
    images = [ctx.train[i]["image"].convert("RGB") for i in picks]

    proposed = [
        ("original", lambda im: im),
        ("horizontal flip", ImageOps.mirror),
        ("rotation +-15", lambda im: aug_rotate(im, rng)),
        ("brightness/contrast/sat 0.2", lambda im: aug_color_jitter(im, rng)),
        ("resized crop scale 0.6-1", lambda im: aug_random_resized_crop(im, rng, scale=(0.6, 1.0))),
    ]
    rejected = [
        ("original", lambda im: im),
        ("hue jitter 0.4", lambda im: aug_hue_shift(im, rng)),
        ("resized crop 0.08-1 (default)", lambda im: aug_random_resized_crop(im, rng)),
        ("resized crop 0.08-1 (again)", lambda im: aug_random_resized_crop(im, rng)),
        ("vertical flip", ImageOps.flip),
    ]
    for name, transforms in (("proposed", proposed), ("rejected", rejected)):
        fig, axes = plt.subplots(len(images), len(transforms),
                                 figsize=(3 * len(transforms), 3 * len(images)), squeeze=False)
        for row, img in zip(axes, images):
            for ax, (title, fn) in zip(row, transforms):
                ax.imshow(fn(img))
                ax.set_title(title, fontsize=9)
                ax.axis("off")
        fig.suptitle(f"Augmentations - {name}")
        save_fig(fig, ctx.fig_dir / f"06_augmentation_{name}.png")

    crops = bird_crop_analysis(ctx)
    if crops is None:
        ctx.summary.pop("augment", None)
    else:
        ctx.summary["augment"] = crops


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
STEPS = {
    "inspect": step_inspect,
    "classes": step_classes,
    "images": step_images,
    "quality": step_quality,
    "captions": step_captions,
    "augment": step_augment,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="EDA for CUB-200-2011 with captions")
    parser.add_argument("--out", default="outputs/eda", help="output folder")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    parser.add_argument("--sample", type=int, default=None, help="use N random images per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=int, default=6, help="max differing dHash bits")
    parser.add_argument("--dup-review", default="configs/eda/near_duplicate_review.csv",
                        help="CSV of by-eye verdicts for near-duplicate pairs")
    parser.add_argument("--cub-meta", default="data/CUB_200_2011",
                        help="folder with the official CUB text files (optional)")
    parser.add_argument("--fetch-cub-meta", action="store_true",
                        help="download the official CUB text files into --cub-meta if missing")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached image stats")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache folder")
    args = parser.parse_args(argv)

    if args.fetch_cub_meta and load_official_metadata(Path(args.cub_meta)) is None:
        fetch_official_metadata(Path(args.cub_meta))
    train, test = load_splits(args.sample, args.seed, args.cache_dir)
    ctx = EDAContext(train=train, test=test, out=Path(args.out), seed=args.seed, sample=args.sample,
                     recompute=args.recompute, near_dup_threshold=args.near_dup_threshold,
                     dup_review=Path(args.dup_review), cub_meta=Path(args.cub_meta))
    for name in args.steps:
        print(f"[eda] step: {name}")
        STEPS[name](ctx)
        ctx.save_summary()
    print(f"[eda] done. Summary: {ctx.out / 'summary.json'}")


if __name__ == "__main__":
    main()
