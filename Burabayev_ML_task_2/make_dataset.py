"""Build the Lab #5 data: UCI Heart Disease (Cleveland), binarised, complete cases only.

Source: UCI Machine Learning Repository, "Heart Disease", file ``processed.cleveland.data``.
    https://archive.ics.uci.edu/static/public/45/heart+disease.zip
The archive's ``heart-disease.names`` asks that publications using the data credit the
investigators: Andras Janosi (Hungarian Institute of Cardiology, Budapest), William
Steinbrunn (University Hospital, Zurich), Matthias Pfisterer (University Hospital, Basel)
and Robert Detrano (V.A. Medical Center, Long Beach, and Cleveland Clinic Foundation).
Original paper: Detrano et al., "International application of a new probability algorithm
for the diagnosis of coronary artery disease", American Journal of Cardiology 64:304-310,
1989. (No licence text is shipped in the archive; check the UCI dataset page before
redistributing beyond the course.)

Transformations, in order:
    1. Parse ``processed.cleveland.data``: 303 rows, 13 features + ``num`` (0..4),
       missing values written as ``?``.
    2. Drop the rows with any missing value. They are all in ``ca`` or ``thal``;
       imputation is Week 2's topic and would distract from the trees.
    3. Target ``disease`` = 1 if ``num > 0`` else 0 (angiographic narrowing present),
       the binarisation used by the ML papers on this database. ``num`` is dropped:
       it *is* the target.
    4. Columns keep the UCI names and the UCI numeric codes; the handout documents them.

Output (next to this script):
    data/heart_disease.csv     13 features, disease   -> students

The file is one labelled table, not a train/test pair: students make their own split.

Run:
    python make_dataset.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

UCI_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
MEMBER = "processed.cleveland.data"

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "disease"

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"


def fetch_raw(url: str = UCI_URL) -> pd.DataFrame:
    """Download the UCI archive and parse the processed Cleveland file.

    Parameters
    ----------
    url : str
        Location of the UCI Heart Disease zip archive.

    Returns
    -------
    pandas.DataFrame
        303 rows; the 13 columns of ``FEATURES`` plus ``num``. ``?`` is parsed as NaN.
    """
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_bytes = archive.read(MEMBER)
    return pd.read_csv(io.BytesIO(csv_bytes), header=None, names=FEATURES + ["num"], na_values="?")


def build_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep complete cases and replace the 0..4 ``num`` by the binary ``disease``.

    Parameters
    ----------
    raw : pandas.DataFrame
        Output of :func:`fetch_raw`.

    Returns
    -------
    pandas.DataFrame
        Complete rows; features as in ``FEATURES`` (integers except the float
        ``oldpeak``) and the 0/1 column ``disease``.
    """
    complete = raw.dropna().reset_index(drop=True)
    frame = complete[FEATURES].copy()
    for column in FEATURES:
        if column != "oldpeak":
            assert (frame[column] % 1 == 0).all(), f"{column} has non-integer values"
            frame[column] = frame[column].astype(int)
    frame[TARGET] = (complete["num"] > 0).astype(int)
    return frame


def check_frame(frame: pd.DataFrame) -> None:
    """Assert the invariants the handout and the reference solution rely on.

    Raises
    ------
    AssertionError
        If values are missing, ``num`` leaked into the output, the target is not 0/1,
        or the column order is not ``FEATURES + [TARGET]``.
    """
    assert list(frame.columns) == FEATURES + [TARGET]
    assert not frame.isna().any().any()
    assert set(frame[TARGET].unique()) == {0, 1}


def main() -> None:
    """Fetch the UCI source, build and verify the table, write the CSV."""
    raw = fetch_raw()
    frame = build_frame(raw)
    check_frame(frame)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(DATA_DIR / "heart_disease.csv", index=False)

    print(f"raw rows: {len(raw)}  rows with a missing value dropped: {len(raw) - len(frame)}")
    print(f"kept: {len(frame)} rows, positive rate {frame[TARGET].mean():.4f}")
    print(f"exact duplicate rows in output: {int(frame.duplicated().sum())}")
    print(f"wrote {DATA_DIR / 'heart_disease.csv'}")


if __name__ == "__main__":
    main()
