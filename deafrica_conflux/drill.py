"""
Run a polygon drill step on a scene.

Matthew Alger, Vanessa Newey
Geoscience Australia
2021
"""

import json
import logging
import warnings
from types import ModuleType

import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray
import shapely.geometry
from datacube.model import Dataset
from datacube.utils.geometry import Geometry
from odc.dscache import DatasetCache
from skimage.measure import regionprops

from deafrica_conflux.io import find_geotiff_files
from deafrica_conflux.text import task_id_to_tuple

_log = logging.getLogger(__name__)


def _get_directions(og_geom: shapely.geometry.Polygon, int_geom: shapely.geometry.Polygon) -> set:
    """
    Helper to get direction of intersection between geometry, intersection.

    Arguments
    ---------
    og_geom : shapely.geometry.Polygon
        Original polygon.

    int_geom : shapely.geometry.Polygon
        Polygon after intersecting with extent.

    Returns
    -------
        set of directions in which the polygon overflows the extent.
    """
    boundary_intersections = int_geom.boundary.difference(og_geom.boundary)
    try:
        # Is a MultiLineString.
        boundary_intersection_lines = list(boundary_intersections.geoms)
    except AttributeError:
        # Is not a MultiLineString.
        boundary_intersection_lines = [boundary_intersections]
    # Split up multilines.
    boundary_intersection_lines_ = []
    for line_ in boundary_intersection_lines:
        coords = list(line_.coords)
        for a, b in zip(coords[:-1], coords[1:]):
            boundary_intersection_lines_.append(shapely.geometry.LineString((a, b)))
    boundary_intersection_lines = boundary_intersection_lines_

    boundary_directions = set()
    for line in boundary_intersection_lines:
        angle = np.arctan2(
            line.coords[1][1] - line.coords[0][1], line.coords[1][0] - line.coords[0][0]
        )
        horizontal = abs(angle) <= np.pi / 4 or abs(angle) >= 3 * np.pi / 4

        if horizontal:
            ys_line = [c[1] for c in line.coords]
            southern_coord_line = min(ys_line)
            northern_coord_line = max(ys_line)

            # Find corresponding southernmost/northernmost point
            # in intersection
            try:
                ys_poly = [c[1] for g in list(int_geom.boundary.geoms) for c in g.coords]
            except AttributeError:
                ys_poly = [c[1] for c in int_geom.boundary.coords]
            southern_coord_poly = min(ys_poly)
            northern_coord_poly = max(ys_poly)

            # If the south/north match the south/north, we have the
            # south/north boundary
            if southern_coord_poly == southern_coord_line:
                # We are south!
                boundary_directions.add("South")
            elif northern_coord_poly == northern_coord_line:
                boundary_directions.add("North")
        else:
            xs_line = [c[0] for c in line.coords]
            western_coord_line = min(xs_line)
            eastern_coord_line = max(xs_line)

            # Find corresponding southernmost/northernmost point
            # in intersection
            try:
                xs_poly = [c[0] for g in list(int_geom.boundary.geoms) for c in g.coords]
            except AttributeError:
                xs_poly = [c[0] for c in int_geom.boundary.coords]
            western_coord_poly = min(xs_poly)
            eastern_coord_poly = max(xs_poly)

            # If the south/north match the south/north, we have the
            # south/north boundary
            if western_coord_poly == western_coord_line:
                # We are west!
                boundary_directions.add("West")
            elif eastern_coord_poly == eastern_coord_line:
                boundary_directions.add("East")
    return boundary_directions


def get_intersections(polygons_gdf: gpd.GeoDataFrame, ds_extent: Geometry) -> pd.DataFrame:
    """
    Find which polygons intersect with a Dataset or DataArray extent
    and in what direction.

    Arguments
    ---------
    polygons_gdf : gpd.GeoDataFrame
        Set of polygons.

    ds_extent : Geometry
        Valid extent of a dataset to check intersection against.

    Returns
    -------
    pd.DataFrame
        Table of intersections.
    """
    # Check if the set of polygons and the dataset extent have the same
    # CRS.
    assert polygons_gdf.crs == ds_extent.crs

    # Get the geometry of the dataset extent.
    ds_extent_geom = ds_extent.geom

    all_intersection = polygons_gdf.geometry.intersection(ds_extent_geom)
    # Which ones have decreased in area thanks to our intersection?
    intersects_mask = ~(all_intersection.area == 0)
    ratios = all_intersection.area / polygons_gdf.area
    directions = []
    dir_names = ["North", "South", "East", "West"]
    for ratio, intersects, idx in zip(ratios, intersects_mask, ratios.index):
        # idx is index into gdf
        if not intersects or ratio == 1:
            directions.append({d: False for d in dir_names})
            continue
        og_geom = polygons_gdf.loc[idx].geometry
        # Buffer to dodge some bad geometry behaviour
        int_geom = all_intersection.loc[idx].buffer(0)
        dirs = _get_directions(og_geom, int_geom)
        directions.append({d: d in dirs for d in dir_names})
        assert any(directions[-1].values())
    return pd.DataFrame(directions, index=ratios.index)


def get_polygons_within_ds_extent(polygons_gdf: gpd.GeoDataFrame, ds: Dataset) -> gpd.GeoDataFrame:
    """
    Filter a set of polygons to include only polygons within (contained in)
    the extent of a dataset.
    """
    # Get the extent of the dataset.
    ds_extent = ds.extent
    ds_extent_crs = ds_extent.crs
    ds_extent_geom = ds_extent.geom
    ds_extent_gdf = gpd.GeoDataFrame(geometry=[ds_extent_geom], crs=ds_extent_crs).to_crs(
        polygons_gdf.crs
    )

    # Get all polygons that are contained withn the extent of the dataset.
    polygon_ids_within_ds_extent = ds_extent_gdf.sjoin(
        polygons_gdf, how="inner", predicate="contains"
    )["index_right"].to_list()
    polygons_within_ds_extent = polygons_gdf.loc[polygon_ids_within_ds_extent]

    return polygons_within_ds_extent


def get_polygons_intersecting_ds_extent(
    polygons_gdf: gpd.GeoDataFrame, ds: Dataset
) -> gpd.GeoDataFrame:
    """
    Filter a set of polygons to only include polygons that intersect with
    the extent of a dataset.

    Parameters
    ---------
    polygons_gdf : gpd.GeoDataFrame
    ds : Dataset

    Returns
    -------
    gpd.GeoDataFrame
    """
    # Get the extent of the dataset.
    ds_extent = ds.extent
    ds_extent_crs = ds_extent.crs
    ds_extent_geom = ds_extent.geom
    ds_extent_gdf = gpd.GeoDataFrame(geometry=[ds_extent_geom], crs=ds_extent_crs).to_crs(
        polygons_gdf.crs
    )

    # Get all polygons that intersect with the extent of the dataset.
    polygon_ids_intersecting_ds_extent = ds_extent_gdf.sjoin(
        polygons_gdf, how="inner", predicate="intersects"
    )["index_right"].to_list()
    polygons_intersecting_ds_extent = polygons_gdf.loc[polygon_ids_intersecting_ds_extent]

    return polygons_intersecting_ds_extent


def filter_large_polygons(polygons_gdf: gpd.GeoDataFrame, ds: Dataset) -> gpd.GeoDataFrame:
    """
    Filter out large polygons from the set of polygons.
    Large polygons are defined as polygons which are large than 3 scenes
    in width and in height.

    Arguments
    ---------
    polygons_gdf : gpd.GeoDataFrame
    ds : datacube.model.Dataset

    Returns
    -------
    gpd.GeoDataFrame
    """
    # Get the extent of the dataset.
    ds_extent = ds.extent

    # Reproject the extent of the dataset to match the set of polygons.
    ds_extent = ds_extent.to_crs(polygons_gdf.crs)

    # Get the bounding box of the extent of the dataset.
    bbox = ds_extent.boundingbox
    left, bottom, right, top = bbox

    # Create a polygon 3 dataset extents in width and height.
    width = right - left
    height = top - bottom

    testbox = shapely.geometry.Polygon(
        [
            (left - width, bottom - height),
            (left - width, top + height),
            (right + width, top + height),
            (right + width, bottom - height),
        ]
    )

    filtered_polygons_gdf = polygons_gdf[~polygons_gdf.geometry.intersects(testbox.boundary)]

    return filtered_polygons_gdf


def remove_duplicate_datasets(required_datasets: list[Dataset]) -> list[Dataset]:
    """
    Remove duplicate datasets based on region code and creation date.
    Picks the most recently created dataset.

    Parameters
    ----------
    required_datasets : list
        List of datasets to filter.

    Returns
    -------
    list
        List of filtered datasets.
    """
    filtered_req_datasets = []

    ds_region_codes = list(set([ds.metadata.region_code for ds in required_datasets]))
    for regioncode in ds_region_codes:
        matching_ds = [ds for ds in required_datasets if ds.metadata.region_code == regioncode]
        matching_ds_sorted = sorted(matching_ds, key=lambda x: x.metadata.creation_dt, reverse=True)
        keep = matching_ds_sorted[0]
        filtered_req_datasets.append(keep)

    return filtered_req_datasets


def drill(
    plugin: ModuleType,
    task_id_string: str,
    cache: DatasetCache,
    polygon_rasters_split_by_tile_directory: str,
    dc: datacube.Datacube | None = None,
) -> pd.DataFrame:
    """
    Perform a polygon drill.

    Parameters
    ----------
    plugin : ModuleType
        A validated plugin to drill with.
    task_id_string : str
        Task id to run drill on in string format.
    cache : DatasetCache
        Dataset cache to read datasets from.
    polygon_rasters_split_by_tile_directory : str
        Directory to search for polygons raster files.
    dc : datacube.Datacube | None, optional
        Optional existing Datacube., by default None

    Returns
    -------
    Drill table : pd.DataFrame
        Index = polygon ID
        Columns = output bands
    """
    # TODO: Generalize to work with multiple products and
    # products with multiple measurements.

    # Check the plugin does not have multiple products/measurements to load.
    # Using multiple products/measurements is not is not implemented.
    measurements = plugin.measurements
    if len(measurements) > 1:
        raise NotImplementedError("Expected 1 measurement in plugin")
    else:
        measurement = measurements[0]

    # Get a datacube if we don't have one already.
    if dc is None:
        dc = datacube.Datacube(app="deafrica-conflux-drill")

    # Get the gridspec/grid name from the cache.
    grid = cache.get_info_dict("stats/config")["grid"]
    _log.debug(f"Found grid {grid}")

    # Parse the task id tuple from the task string.
    task_id_tuple = task_id_to_tuple(task_id_string)

    # Find the polygons raster tile to use for the task.
    _log.info("Finding polygon raster tile....")
    search_pattern = f".*x{task_id_tuple[1]:03d}_y{task_id_tuple[2]:03d}.*"
    found_raster_tile = find_geotiff_files(
        path=polygon_rasters_split_by_tile_directory, pattern=search_pattern
    )
    # The should only be one polygons raster file  for each tile.
    assert len(found_raster_tile) == 1

    # Load the enumerated waterbodies polygons raster tile.
    polygons_raster = rioxarray.open_rasterio(found_raster_tile[0]).squeeze("band", drop=True)
    _log.info(f"Loaded {found_raster_tile[0]}")
    assert polygons_raster.geobox.crs == plugin.output_crs
    assert polygons_raster.geobox.resolution == plugin.resolution

    # Get the datasets for the task.
    required_datasets = [ds for ds in cache.stream_grid_tile(task_id_tuple, grid)]

    # Create a datacube query object to use to load the found datasets.
    query = {}

    if hasattr(plugin, "resampling"):
        query["resampling"] = plugin.resampling
    else:
        query["resampling"] = "nearest"

    if hasattr(plugin, "dask_chunks"):
        query["dask_chunks"] = plugin.dask_chunks

    query["group_by"] = "solar_day"

    query["like"] = polygons_raster.geobox

    query["measurements"] = [measurement]

    query["datasets"] = required_datasets

    _log.info(f"Query object to use for loading data {json.dumps(query)}")

    # Load the datasets.
    ds_ = dc.load(**query)

    # TODO: Take care of empty regions or other errors resulting in empty dataframes.
    # if len(filtered_polygons_gdf) == 0:

    # Transform the loaded data.
    # Force warnings to raise exceptions.
    # This means users have to explicitly ignore warnings.
    ds = ds_.isel(time=0)

    with warnings.catch_warnings():
        warnings.filterwarnings("error")
        ds_transformed = plugin.transform(ds)[measurement]

    # For each polygon, perform the summary.
    props = regionprops(
        label_image=polygons_raster.values,
        intensity_image=ds_transformed.values,
        extra_properties=(plugin.summarise,),
    )

    summary_df_list = []
    for region_prop in props:
        polygon_summary_df = region_prop.summarise
        polygon_summary_df.index = [region_prop.label]
        summary_df_list.append(polygon_summary_df)

    summary_df = pd.concat(summary_df_list, ignore_index=False)

    return summary_df
