"""Stack Parquet scene outputs into other formats.

Lots of this code is domain-specific and not intended to be fully general.

Matthew Alger
Geoscience Australia
2021
"""

import collections
import logging
import os
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from tqdm import tqdm

from deafrica_conflux.io import check_dir_exists, check_if_s3_uri, load_parquet_file

_log = logging.getLogger(__name__)


def update_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the data in the timeseries DataFrame.

    Arguments
    ---------
    df : pd.DataFrame
        The polygon base timeseries result.

    Returns
    -------
    pd.DataFrame
        The polygon base timeseries.
    """
    assert "date" in df.columns

    keep = np.unique(df["date"])

    # Group by day
    df = df.groupby(pd.Grouper(key="date", axis=0, freq="D")).sum()
    # Remove filler rows.
    df = df[df.index.isin(keep)]

    df["pc_wet"] = (df["px_wet"] / df["px_total"]) * 100.0
    df["pc_dry"] = (df["px_dry"] / df["px_total"]) * 100.0
    df["pc_invalid"] = (df["px_invalid"] / df["px_total"]) * 100.0

    # If the proportion of the waterbody missing is greater than 10%,
    # set the values for pc_wet and pc_dry to nan.
    df.loc[df["pc_invalid"] > 10.0, "pc_wet"] = np.nan
    df.loc[df["pc_invalid"] > 10.0, "pc_dry"] = np.nan

    df.sort_index(inplace=True)
    return df


def stack_waterbodies_parquet_to_csv(
    parquet_file_paths: [str | Path],
    output_directory: str | Path,
    polygons_ids_mapping: dict[str, str],
):
    """
    Stack Parquet files into CSVs.

    Arguments
    ---------
    parquet_file_paths : [str | Path]
        List of paths to Parquet files to stack.

    output_directory : str | Path
        Path to output directory.

    polygons_ids_mapping: dict[str, str]
        Dictionary mapping numerical polygon ids (WB_ID) to string polygon ids (UID)
    """
    # "Support" pathlib Paths.
    parquet_file_paths = [str(pq_file_path) for pq_file_path in parquet_file_paths]
    output_directory = str(output_directory)

    assert polygons_ids_mapping

    # id -> [series of date x bands]
    id_to_series = collections.defaultdict(list)
    n_total = len(parquet_file_paths)
    label = f"Reading {n_total} drill output parquet files."
    with tqdm(parquet_file_paths, desc=label, total=n_total) as parquet_file_paths:
        for pq_file_path in parquet_file_paths:
            df = load_parquet_file(pq_file_path)
            # df is ids x bands
            # for each ID...
            for uid, series in df.iterrows():
                series.name = series.date
                uid = polygons_ids_mapping[str(uid)]
                id_to_series[uid].append(series)

    n_total = len(id_to_series.items())
    for i, item in enumerate(id_to_series.items()):
        uid, seriess = item
        _log.info(f"Writing csv file for polygon {uid} ({i + 1}/{n_total})")

        df = pd.DataFrame(seriess)
        df = update_timeseries(df=df)

        output_file_parent_directory = os.path.join(output_directory, f"{uid[:4]}")
        output_file_path = os.path.join(output_file_parent_directory, f"{uid}.csv")

        if check_if_s3_uri(output_file_parent_directory):
            fs = fsspec.filesystem("s3")
        else:
            fs = fsspec.filesystem("file")

        if not check_dir_exists(output_file_parent_directory):
            fs.mkdirs(output_file_parent_directory, exist_ok=True)
            _log.info(f"Created directory: {output_file_parent_directory}")

        with fs.open(output_file_path, "w") as f:
            df.to_csv(f, index_label="date")

        _log.info(f"CSV file written to {output_file_path}")
