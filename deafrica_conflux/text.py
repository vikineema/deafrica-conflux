"""Text formatting functions"""
import os
from datetime import datetime


def date_to_stack_format_str(date: datetime) -> str:
    """
    Format a date to match DE Africa conflux products datetime.

    Arguments
    ---------
    date : datetime

    Returns
    -------
    str
    """
    # e.g. 1987-05-24T01:30:18Z
    return date.strftime("%Y-%m-%dT%H:%M:%SZ")


def serialise_date(date: datetime) -> str:
    """
    Serialise a date.

    Arguments
    ---------
    date : datetime

    Returns
    -------
    str
    """
    return date.strftime("%Y%m%d-%H%M%S-%f")


def unserialise_date(date: str) -> datetime:
    """
    Unserialise a date.

    Arguments
    ---------
    date : str

    Returns
    -------
    datetime
    """
    return datetime.datetime.strptime(date, "%Y%m%d-%H%M%S-%f")


def date_to_day_str(date: datetime) -> str:
    """
    Serialise a date discarding hours/mins/seconds.

    Arguments
    ---------
    date : datetime

    Returns
    -------
    str
    """
    return date.strftime("%Y%m%d")


def make_parquet_file_name(drill_name: str, uuid: str, centre_date: datetime) -> str:
    """
    Make filename for Parquet.

    Arguments
    ---------
    drill_name : str
        Name of the drill.

    uuid : str
        UUID of reference dataset.

    centre_date : datetime
        Centre date of reference dataset.

    Returns
    -------
    str
        Parquet filename.
    """
    datestring = serialise_date(centre_date)

    parquet_file_name = f"{drill_name}_{uuid}_{datestring}.pq"

    return parquet_file_name


def parse_tile_ids(file_path: str) -> str:
    """
    Parse tile ids from a file path.

    Parameters
    ----------
    file_path : str
        File path to get the tile id from.

    Returns
    -------
    str
        Tile id
    """
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    x_id = int(file_name.split("_")[0].lstrip("x"))
    y_id = int(file_name.split("_")[1].lstrip("y"))
    tile_id = (x_id, y_id)
    return tile_id


def task_id_to_string(task_id_tuple: tuple) -> str:
    """
    Transform a task id tuple to a string.

    Parameters
    ----------
    task_id_tuple : tuple
        Task id as a tuple.

    Returns
    -------
    str
        Task id as string.
    """
    period, x, y = task_id_tuple

    task_id_string = f"{period}/{x:02d}/{y:02d}"

    return task_id_string


def task_id_to_tuple(task_id_string: str) -> tuple:
    """
    Transform a task id string to a tuple.

    Parameters
    ----------
    task_id_string : str
        Task id as string.

    Returns
    -------
    tuple
        Task id as a tuple.
    """
    sep = "/" if "/" in task_id_string else ","

    period, x, y = task_id_string.split(sep)

    if period.startswith("x"):
        period, x, y = y, period, x

    x = int(x)
    y = int(y)

    task_id_tuple = (period, x, y)

    return task_id_tuple
