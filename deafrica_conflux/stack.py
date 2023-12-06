"""Stack Parquet scene outputs into other formats.

Lots of this code is domain-specific and not intended to be fully general.

Matthew Alger
Geoscience Australia
2021
"""

import logging
import os
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd

from deafrica_conflux.io import (
    check_dir_exists,
    check_if_s3_uri,
    find_parquet_files,
    load_parquet_file,
)

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
    polygon_id: str, drill_output_directory: str | Path, output_directory: str | Path
):
    # Support pathlib paths.
    drill_outputs_directory = str(drill_output_directory)
    output_directory = str(output_directory)

    _log.info(f"Stacking timeseries for polygon {polygon_id}")
    # Find drill output files for the polygon.
    pq_files = find_parquet_files(
        path=drill_outputs_directory, pattern=f".*{polygon_id}.*", verbose=False
    )

    if not pq_files:
        _log.info(f"Found 0 drill output parquet files for polygon {polygon_id}")
    else:
        _log.info(f"Found {len(pq_files)} drill output parquet files for polygon {polygon_id}")
        df_list = []
        for file in pq_files:
            file_df = load_parquet_file(file)
            df_list.append(file_df)
        df = pd.concat(df_list, ignore_index=False)
        df = df.set_index(df["date"])
        df = update_timeseries(df)

        output_file_parent_directory = os.path.join(output_directory, f"{polygon_id[:4]}")
        output_file_path = os.path.join(output_file_parent_directory, f"{polygon_id}.csv")

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
