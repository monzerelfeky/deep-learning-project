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
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures without needing a display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

TRAIN_REPO = "Multimodal-Fatima/CUB_train"
TEST_REPO = "Multimodal-Fatima/CUB_test"
STATS_SIZE = 256  # pixel statistics are computed on images resized to 256x256
# Largest per-pixel max-min channel spread (0-255) below this = no color anywhere = grayscale.
# A mean over the image fails both ways: color birds on big gray sky/water score low, and a
# black-and-white photo with one colored eye scores lower than a fully gray one
GRAY_MAX_CHROMA = 10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


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
    near_dup_threshold: int = 4
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
# Entry point
# --------------------------------------------------------------------------
STEPS = {
    "inspect": step_inspect,
    "classes": step_classes,
    "images": step_images,
}


def main(argv=None):
    parser = argparse.ArgumentParser(description="EDA for CUB-200-2011 with captions")
    parser.add_argument("--out", default="outputs/eda", help="output folder")
    parser.add_argument("--steps", nargs="+", choices=list(STEPS), default=list(STEPS))
    parser.add_argument("--sample", type=int, default=None, help="use N random images per split")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-dup-threshold", type=int, default=4, help="max differing dHash bits")
    parser.add_argument("--recompute", action="store_true", help="ignore the cached image stats")
    parser.add_argument("--cache-dir", default=None, help="Hugging Face cache folder")
    args = parser.parse_args(argv)

    train, test = load_splits(args.sample, args.seed, args.cache_dir)
    ctx = EDAContext(train=train, test=test, out=Path(args.out), seed=args.seed, sample=args.sample,
                     recompute=args.recompute, near_dup_threshold=args.near_dup_threshold)
    for name in args.steps:
        print(f"[eda] step: {name}")
        STEPS[name](ctx)
        ctx.save_summary()
    print(f"[eda] done. Summary: {ctx.out / 'summary.json'}")


if __name__ == "__main__":
    main()
