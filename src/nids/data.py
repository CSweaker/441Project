"""Data loading and label engineering helpers."""

from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd

from .config import ATTACK_FAMILY_MAP, NSL_KDD_COLUMNS


ALLOWED_DATA_FILES = {
    "kddtrain+.txt": "KDDTrain+.txt",
    "kddtest+.txt": "KDDTest+.txt",
}


def extract_zip_if_needed(
    zip_path: str | Path,
    data_dir: str | Path = "data",
    *,
    overwrite: bool = False,
) -> Path:
    """Safely extract only official NSL-KDD train/test text files.

    Earlier project drafts extracted every .txt/.csv/.py file in a zip. This version
    is intentionally stricter: it only accepts KDDTrain+.txt and KDDTest+.txt, writes
    them to canonical names, and avoids silently overwriting existing files unless
    ``overwrite=True`` is passed.
    """
    zip_path = Path(zip_path)
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_name = Path(member.filename).name
            canonical_name = ALLOWED_DATA_FILES.get(member_name.lower())
            if canonical_name is None:
                continue

            target = data_dir / canonical_name
            if target.exists() and not overwrite:
                extracted.append(target)
                continue
            with zf.open(member) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.append(target)

    if not extracted:
        allowed = ", ".join(sorted(ALLOWED_DATA_FILES.values()))
        raise FileNotFoundError(
            f"No NSL-KDD train/test files found in {zip_path}. Expected one of: {allowed}."
        )
    return data_dir


def load_nsl_kdd(path: str | Path, has_header: bool = False) -> pd.DataFrame:
    """Load a NSL-KDD .txt/.csv file and return a DataFrame with standard columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, names=NSL_KDD_COLUMNS)

    # Some public mirrors omit the difficulty column. Add it if needed.
    if list(df.columns) != NSL_KDD_COLUMNS:
        if df.shape[1] == 41:
            feature_cols = NSL_KDD_COLUMNS[:-2]
            df.columns = feature_cols
            df["label"] = "unknown"
            df["difficulty"] = 0
        elif df.shape[1] == 42:
            df.columns = NSL_KDD_COLUMNS[:-1]
            df["difficulty"] = 0
        elif df.shape[1] == 43:
            df.columns = NSL_KDD_COLUMNS
        else:
            raise ValueError(
                f"Unexpected number of columns in {path}: {df.shape[1]}. "
                "Expected 41, 42, or 43 columns."
            )

    df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary and five-class labels used by the two-stage IDS pipeline."""
    out = df.copy()
    out["binary_target"] = (out["label"] != "normal").astype(int)
    out["attack_family"] = out["label"].map(ATTACK_FAMILY_MAP).fillna(
        out["label"].where(out["label"].eq("normal"), "Other")
    )
    out["final_target"] = out["attack_family"].where(out["label"] != "normal", "normal")
    return out


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the input feature columns, excluding labels and difficulty."""
    drop_cols = ["label", "difficulty", "binary_target", "attack_family", "final_target"]
    return df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
