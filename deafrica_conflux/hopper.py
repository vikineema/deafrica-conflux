"""Produce datasets for consumption.

Matthew Alger, Alex Leith
Geoscience Australia
2021
"""

import logging
from types import SimpleNamespace
from typing import Dict, Iterable

import toolz
from datacube import Datacube
from datacube.model import Dataset
from odc.stats.tasks import CompressedDataset, compress_ds
from odc.stats.utils import Cell

_log = logging.getLogger(__name__)
dt_range = SimpleNamespace(start=None, end=None)


def find_datasets(
    query: Dict[str, str], products: [str], limit: int = None, dc: Datacube = None
) -> Iterable[Dataset]:
    """
    Find datasets with a Datacube query.

    Heavily based on datacube_alchemist.worker._find_datsets.

    Arguments
    ---------
    query : Dict[str, str]

    products : [str]
        List of products to search.

    limit : int
        Maximum number of datasets to return (default unlimited).

    dc : Datacube
        Datacube or None.

    Returns
    -------
    Generator of datasets
    """
    if dc is None:
        dc = Datacube()

    # Find many datasets across many products with a limit
    count = 0

    for product in products:
        datasets = dc.index.datasets.search(
            product=product,
            **query,
        )

        try:
            for dataset in datasets:
                yield dataset
                count += 1
                if limit is not None and count >= limit:
                    return
        except ValueError as error:
            _log.warning(
                f"Error searching for datasets. Maybe none were returned? Error was {error}"
            )
            continue


def check_ds_region(region_codes: list, ds: Dataset) -> str:
    """
    Check if the region code of a dataset is in the list of region codes
    provided. If True, returns the dataset's ID, if False, returns an empty
    string.

    Parameters
    ----------
    region_codes : list
        A list of region codes to check the dataset's region code against.
    ds : Dataset
        The dataset to check.

    Returns
    -------
    str
        If the dataset's region code is in the list of region codes `region_codes`,
        returns the dataset's ID.
        If it is not, returns an empty string.
    """
    # Get the dataset's region code.
    ds_region_code = ds.metadata.region_code
    if ds_region_code in region_codes:
        # Get the dataset id.
        ds_id = str(ds.id)
        return ds_id
    else:
        return ""


# From https://github.com/opendatacube/odc-stats/blob/develop/odc/stats/tasks.py
def update_start_end(x, out):
    if out.start is None:
        out.start = x
        out.end = x
    else:
        out.start = min(out.start, x)
        out.end = max(out.end, x)


def persist(ds: Dataset) -> CompressedDataset:
    _ds = compress_ds(ds)
    update_start_end(_ds.time, dt_range)
    return _ds


def bin_solar_day(
    cells: dict[tuple[int, int], Cell]
) -> dict[tuple[str, int, int], list[CompressedDataset]]:
    """
    Bin by solar day.
    :param cells: (x,y) -> Cell(dss: List[CompressedDataset], geobox: GeoBox, idx: Tuple[int, int])
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
