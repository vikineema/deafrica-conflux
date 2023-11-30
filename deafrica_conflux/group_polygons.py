import logging

import geopandas as gpd

from deafrica_conflux.id_field import guess_id_field

_log = logging.getLogger(__name__)


def get_intersecting_polygons_ids(
    region: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame, use_id=""
) -> list:
    """
    Get the IDs of the polygons that intersect with a region.

    Parameters
    ----------
    region : gpd.GeoDataFrame
        A GeoDataFrame of the region of interest.
    polygons_gdf : gpd.GeoDataFrame
        A set of polygons to filter by intersection with the region.
    use_id : str
        Column to use as the id field for the polygons to filter.
    Returns
    -------
    list
        A list of the ids of the polygons that intersect with the region.
    """
    if region.crs != polygons_gdf.crs:
        region = region.to_crs(polygons_gdf.crs)

    assert region.crs == polygons_gdf.crs

    if not use_id:
        intersecting_polygons_ids = gpd.sjoin(
            polygons_gdf, region, how="inner", predicate="intersects"
        ).index.to_list()
    else:
        # Verify the values in the use_id column are unique.
        id_field = guess_id_field(input_gdf=polygons_gdf, use_id=use_id)

        intersecting_polygons_ids = gpd.sjoin(
            polygons_gdf, region, how="inner", predicate="intersects"
        )[id_field].to_list()

    return intersecting_polygons_ids


def get_intersecting_polygons(
    region: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame, use_id=""
) -> gpd.GeoDataFrame:
    """
    Get the polygons that intersect with a region.

    Parameters
    ----------
    region : gpd.GeoDataFrame
        A GeoDataFrame of the region of interest.
    polygons_gdf : gpd.GeoDataFrame
        A set of polygons to filter by intersection with the region.
    use_id : str
        Column to use as the id field for the polygons to filter.
    Returns
    -------
    gpd.GeoDataFrame
        Polygons that intersect with the region.
    """
    intersecting_polygons_ids = get_intersecting_polygons_ids(
        region=region, polygons_gdf=polygons_gdf, use_id=use_id
    )

    if not use_id:
        intersecting_polygons = polygons_gdf[polygons_gdf.index.isin(intersecting_polygons_ids)]
    else:
        intersecting_polygons = polygons_gdf[polygons_gdf[use_id].isin(intersecting_polygons_ids)]

    return intersecting_polygons
