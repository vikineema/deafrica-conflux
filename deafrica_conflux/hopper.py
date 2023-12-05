"""Produce datasets for consumption.

Matthew Alger, Alex Leith
Geoscience Australia
2021
"""

import logging
from datetime import datetime
from types import SimpleNamespace

import toolz
from datacube.model import Dataset
from odc.stats.tasks import CompressedDataset, compress_ds
from odc.stats.utils import Cell

_log = logging.getLogger(__name__)
dt_range = SimpleNamespace(start=None, end=None)


# From https://github.com/opendatacube/odc-stats/blob/develop/odc/stats/tasks.py
def update_start_end(x: datetime, out: SimpleNamespace):
    """
    Add a start and end datetime to a CompressedDataset.

    Parameters
    ----------
    x : datetime
        Datetime or date range to use.
    out : SimpleNamespace
        An empty simplenamespace object to fill.
    """
    if out.start is None:
        out.start = x
        out.end = x
    else:
        out.start = min(out.start, x)
        out.end = max(out.end, x)


def persist(ds: Dataset) -> CompressedDataset:
    """
    Mapping function to use when binning a dataset stream.
    """
    # Convert the dataset to a CompressedDataset.
    _ds = compress_ds(ds)
    # Add a start and end datetime to the CompressedDataset.
    update_start_end(_ds.time, dt_range)
    return _ds


def bin_solar_day(
    cells: dict[tuple[int, int], Cell]
) -> dict[tuple[str, int, int], list[CompressedDataset]]:
    """
    Bin by solar day.

    Parameters
    ----------
    cells : dict[tuple[int, int], Cell]
        Cells to bin.

    Returns
    -------
    dict[tuple[str, int, int], list[CompressedDataset]]
        Input cells with datasets binned by day.
    """
    tasks = {}
    for tidx, cell in cells.items():
        # This is a great pylint warning, but doesn't apply here because we
        # only call the lambda from inside each iteration of the loop
        # pylint:disable=cell-var-from-loop
        utc_offset = cell.utc_offset
        grouped = toolz.groupby(lambda ds: (ds.time + utc_offset).date(), cell.dss)

        for day, dss in grouped.items():
            temporal_k = (day.strftime("%Y-%m-%d"),)
            tasks[temporal_k + tidx] = dss

    return tasks
