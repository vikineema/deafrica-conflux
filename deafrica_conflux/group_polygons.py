import geopandas as gpd


def get_intersecting_polygons_ids(region: gpd.GeoDataFrame, polygons_gdf: gpd.GeoDataFrame) -> list:
    """
    Get the IDs of the polygons that intersect with a region.

    Parameters
    ----------
    region : gpd.GeoDataFrame
        A GeoDataFrame of the region of interest.
    polygons_gdf : gpd.GeoDataFrame
        A set of polygons to filter by intersection with the region.

    Returns
    -------
    list
        A list of the ids of the polygons that intersect with the region.
    """

    intersecting_polygons_ids = gpd.sjoin(
        polygons_gdf, region, how="inner", predicate="intersects"
    ).index.to_list()

    return intersecting_polygons_ids
