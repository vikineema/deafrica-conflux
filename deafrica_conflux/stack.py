"""Stack Parquet scene outputs into other formats.

Lots of this code is domain-specific and not intended to be fully general.

Matthew Alger
Geoscience Australia
2021
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import dask
import dask.dataframe as dd
import fsspec
import numpy as np
import pandas as pd
import pyarrow.fs
from odc.stats.model import DateTimeRange

from deafrica_conflux.io import check_dir_exists, check_if_s3_uri, find_parquet_files

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


def extract_date_from_filename(filename) -> datetime:
    """
    Extract a date in the in the format YYYY-MM-DD from
    a file name.

    Parameters
    ----------
    filename : str
        File name to exctract date from.

    Returns
    -------
    datetime
        Extracted date as a datetime object
    """
    # Define a regex pattern to extract dates in the format YYYY-MM-DD
    date_pattern = r"(\d{4}-\d{2}-\d{2})"

    # Search for the date pattern in the filename
    # Note: re.search() finds the first match only of a pattern
    match = re.search(date_pattern, filename)

    if match:
        # Extract the matched date
        return datetime.strptime(match.group(), "%Y-%m-%d")
    else:
        return None


def filter_files_by_date_range(file_paths: list[str], temporal_range: str) -> list[str]:
    """
    Filter file paths using a date range.

    Parameters
    ----------
    file_paths : list[str]
        List of files to filter.
    temporal_range : str
        Date range to filter by e.g. 2023-01--P1M filters file by
        the date range 2023-01-01 to 2023-01-31.

    Returns
    -------
    list[str]
        List of files within the defined temporal range.
    """
    temporal_range_ = DateTimeRange(temporal_range)
    start_date = temporal_range_.start
    end_date = temporal_range_.end

    # Filter files based on the dates extracted from filenames
    filtered_file_paths = []
    for path in file_paths:
        file_date = extract_date_from_filename(os.path.basename(path))

        if file_date:
            # Check if the file date is within the specified date range
            if start_date <= file_date <= end_date:
                filtered_file_paths.append(path)

    return filtered_file_paths


def stack_polygon_timeseries_to_csv(
    polygon_uid: str,
    polygon_stringids_to_tileids: dict[str, list[str]],
    drill_output_directory: str | Path,
    output_directory: str | Path,
    temporal_range: None | str = None,
) -> list[str]:
    """
    Stack the timeseries for a polygon from the drill output parquet files
    into a csv file.

    Parameters
    ----------
    polygon_uid : str
        Unique id for a polygon.
    polygon_stringids_to_tileids: dict[str, list[str]]
        Dictionary mapping the unique polygon ids to the tile ids for the
        tiles they intersect with.
    drill_output_directory : str | Path
        Directory containing the parquet files i.e. outputs from the drill step.
    output_directory : str | Path
        Directory to write the csv files to.
    temporal_range:  None | str = None
        Limit the timeseries for a polygon to only a specific time range.
        e.g. 2023-01--P1M stacks the polygon timeseries for the date
        range 2023-01-01 to 2023-01-31.
    Returns
    -------
    str
        File path of the polygon timeseries csv file.
    """
    _log.info(f"Stacking timeseries for the polygon {polygon_uid}")

    # Support pathlib paths.
    drill_output_directory = str(drill_output_directory)
    output_directory = str(output_directory)

    # Get the file system to use to write.
    if check_if_s3_uri(output_directory):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    # Find all the drill output files.
    intersecting_tile_ids = polygon_stringids_to_tileids.get(polygon_uid)

    if intersecting_tile_ids:
        pattern = "|".join([f".*{i}.*" for i in intersecting_tile_ids])
        drill_output_files = find_parquet_files(
            path=drill_output_directory, pattern=pattern, verbose=True
        )
        if drill_output_files:
            if temporal_range is not None:
                drill_output_files = filter_files_by_date_range(
                    file_paths=drill_output_files, temporal_range=temporal_range
                )
                if not drill_output_files:
                    _log.error(
                        f"No drill output parquet files found for polygon {polygon_uid} for the temporal range {temporal_range} in the folder {drill_output_directory}"
                    )
                    return None

            # Read the parquet files.
            df = pd.read_parquet(
                [f.lstrip("s3://") for f in drill_output_files],
                filesystem=pyarrow.fs.FileSystem.from_uri(drill_output_files[0])[0],
            )

            # Get the timeseries for the specific polygon from the larger table.
            polygon_timeseries = update_timeseries(df.loc[polygon_uid])

            # Write the polygon timeseries to a csv file.
            output_file_parent_directory = os.path.join(output_directory, f"{polygon_uid[:4]}")
            output_file_path = os.path.join(output_file_parent_directory, f"{polygon_uid}.csv")

            # Get the file system to use to write.
            if check_if_s3_uri(output_file_parent_directory):
                fs = fsspec.filesystem("s3")
            else:
                fs = fsspec.filesystem("file")

            if not check_dir_exists(output_file_parent_directory):
                fs.mkdirs(output_file_parent_directory, exist_ok=True)
                _log.info(f"Created directory: {output_file_parent_directory}")

            with fs.open(output_file_path, "w") as f:
                polygon_timeseries.to_csv(f, index_label="date")

            _log.info(f"CSV file written to {output_file_path}")

            return output_file_path

        else:
            _log.error(
                f"No drill output parquet files found for polygon: {polygon_uid} in the folder {drill_output_directory}"
            )
            return None
    else:
        _log.error(f"No intersecting tiles found for polygon: {polygon_uid}")
        return None


def stack_polygon_timeseries_to_csv_old_version(
    polygon_ids: list[str], drill_output_directory: str | Path, output_directory: str | Path
) -> list[str]:
    """
    Stack the timeseries for a polygon from the drill output parquet files
    into a csv file. Best used in parallel processing.

    Parameters
    ----------
    polygon_ids : list[str]
        Unique ids for the polygons.
    drill_output_directory : str | Path
        Directory containing the parquet files i.e. outputs from the drill step.
    output_directory : str | Path
        Directory to write the csv files to.

    Returns
    -------
    list[str]
        File paths of the polygon(s) timeseries csv files.
    """
    # Support pathlib paths.
    drill_output_directory = str(drill_output_directory)
    output_directory = str(output_directory)

    # Get the file system to use to write.
    if check_if_s3_uri(output_directory):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    _log.info(f"Stacking timeseries for polygons: {', '.join(polygon_ids)}")

    # Find all the drill output files.
    drill_output_files = find_parquet_files(path=drill_output_directory, pattern=".*")

    # Read in all the drill output parquet files using dask dataframes.
    df = dd.read_parquet(drill_output_files)

    # Compute the timeseries for each polygon id.
    to_compute = [df.loc[poly_id] for poly_id in polygon_ids]
    polygons_timeseries = dask.compute(*to_compute)

    # Write the timeseries to csv.
    output_file_paths = []
    for frame in polygons_timeseries:
        polygon_id = frame.index[0]
        polygon_timeseries = update_timeseries(frame)

        output_file_parent_directory = os.path.join(output_directory, f"{polygon_id[:4]}")
        output_file_path = os.path.join(output_file_parent_directory, f"{polygon_id}.csv")

        if not check_dir_exists(output_file_parent_directory):
            fs.mkdirs(output_file_parent_directory, exist_ok=True)
            _log.info(f"Created directory: {output_file_parent_directory}")

        with fs.open(output_file_path, "w") as f:
            polygon_timeseries.to_csv(f, index_label="date")

        _log.info(f"CSV file written to {output_file_path}")

        output_file_paths.append(output_file_path)
    return output_file_paths
