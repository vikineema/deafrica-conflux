import logging
import os

import click
import fsspec
import geopandas as gpd
import numpy as np
import rasterio.features
from datacube.utils.cog import to_cog
from datacube.utils.dask import save_blob_to_file, save_blob_to_s3
from datacube.utils.geometry import Geometry
from odc.dscache.tools.tiling import parse_gridspec_with_name
from odc.geo.geobox import GeoBox
from odc.geo.xr import wrap_xr

from deafrica_conflux.cli.logs import logging_setup
from deafrica_conflux.filter_polygons import get_intersecting_polygons
from deafrica_conflux.id_field import guess_id_field
from deafrica_conflux.io import check_dir_exists, check_file_exists, check_if_s3_uri


@click.command(
    "rasterise-polygons",
    no_args_is_help=True,
    help="Rasterize a set of waterbodies polygons by tile.",
)
@click.option("-v", "--verbose", count=True)
@click.option("--grid-name", type=str, help="Grid name africa_{10|20|30|60}", default="africa_30")
@click.option(
    "--product",
    type=str,
    help="Datacube product to get tiles for from the grid.",
    default="wofs_ls",
)
@click.option(
    "--polygons-file-path",
    type=str,
    help="Path to the polygons to be rasterised.",
)
@click.option(
    "--use-id",
    type=str,
    default="WB_ID",
    help="Optional. Unique key id polygons vector file.",
)
@click.option("output-directory", type=str, help="Directory to write the tiled polygon rasters to.")
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    help="Overwrite existing polygons raster file.",
)
def rasterise_polyongs(
    verbose,
    grid_name,
    product,
    polygons_file_path,
    use_id,
    output_directory,
    overwrite,
):
    # Set up logger.
    logging_setup(verbose)
    _log = logging.getLogger(__name__)

    # Support pathlib Paths.
    polygons_file_path = str(polygons_file_path)
    output_directory = str(output_directory)

    # Create the output directory if it does not exist.
    is_s3 = check_if_s3_uri(output_directory)
    if is_s3:
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    if not check_dir_exists(output_directory):
        fs.makedirs(output_directory, exist_ok=True)
        _log.info(f"Created directory {output_directory}")

    tiles_output_directory = os.path.join(output_directory, "product_tiles")
    if not check_dir_exists(tiles_output_directory):
        fs.makedirs(tiles_output_directory, exist_ok=True)
        _log.info(f"Created directory {tiles_output_directory}")

    rasters_output_directory = os.path.join(output_directory, "historical_extent_rasters")
    if not check_dir_exists(rasters_output_directory):
        fs.mkdirs(rasters_output_directory, exist_ok=True)
        _log.info(f"Created directory {output_directory}")

    # Get the GridSpec.
    grid, gridspec = parse_gridspec_with_name(grid_name)
    _log.info(f"Using the grid {grid} with {gridspec}")

    # From the GridSpec get the crs and resolution.
    crs = gridspec.crs
    resolution = abs(gridspec.resolution[0])

    # Read the product footprint.
    product_footprint = gpd.read_file(
        f"https://explorer.digitalearth.africa/api/footprint/{product}"
    ).to_crs(crs)
    # Get the product footprint geopolygon.
    product_footprint = Geometry(geom=product_footprint.geometry[0], crs=crs)

    # Get the tiles covering the product footprint.
    tiles = gridspec.tiles_from_geopolygon(geopolygon=product_footprint)
    tiles = list(tiles)

    # Get the individual tile geometries.
    tile_geometries = []
    tile_ids = []
    for tile in tiles:
        tile_idx, tile_idy = tile[0]
        tile_geometry = tile[1].extent.geom

        tile_geometries.append(tile_geometry)
        tile_ids.append(f"x{tile_idx:03d}_y{tile_idy:03d}")

    tiles_gdf = gpd.GeoDataFrame(data={"tile_ids": tile_ids, "geometry": tile_geometries}, crs=crs)
    _log.info(f"Tile count: {len(tiles_gdf)}")

    tiles_output_fp = os.path.join(tiles_output_directory, f"{product}_tiles.parquet")
    tiles_gdf.to_parquet(tiles_output_fp)
    _log.info(f"{product} tiles written to {tiles_output_fp}")

    # Load the polygons.
    polygons_gdf = gpd.read_parquet(polygons_file_path).to_crs(crs)
    _log.info(f"Polygon count {len(polygons_gdf)}")

    # Check the id column is unique.
    id_field = guess_id_field(input_gdf=polygons_gdf, use_id=use_id)

    _log.info("Filtering out tiles that do not intersect with any polygon...")
    filtered_tiles_gdf = get_intersecting_polygons(
        region=polygons_gdf, polygons_gdf=tiles_gdf, use_id="tile_ids"
    )
    _log.info(f"Filtered out {len(tiles_gdf) - len(filtered_tiles_gdf)} tiles.")
    _log.info(f"Filtered tiles count: {len(filtered_tiles_gdf)}.")

    # Split each row in the tiles into a GeoDataFrame of its own.
    tiles = np.array_split(filtered_tiles_gdf, len(filtered_tiles_gdf))
    assert len(tiles) == len(filtered_tiles_gdf)

    for i, tile in enumerate(tiles):
        tile_id = tile["tile_ids"].iloc[0]
        tile_geometry = tile.geometry.iloc[0]
        _log.info(f"Rasterizing polygons for tile {tile_id} ({i + 1}/{len(tiles)})")

        tile_raster_fp = os.path.join(rasters_output_directory, f"{tile_id}.tif")

        if not overwrite:
            _log.info(f"Checking existence of {tile_raster_fp}")
            exists = check_file_exists(tile_raster_fp)

        if overwrite or not exists:
            # Get the geobox for the region.
            tile_geobox = GeoBox.from_geopolygon(
                geopolygon=Geometry(geom=tile_geometry, crs=crs),
                resolution=resolution,
                crs=crs,
            )

            # Get the polygons that intersect with the tile.
            tile_polygons = get_intersecting_polygons(
                region=tile, polygons_gdf=polygons_gdf, use_id=id_field
            )

            # Rasterise shapes into a numpy array
            shapes = zip(tile_polygons.geometry, tile_polygons[id_field])
            tile_raster_np = rasterio.features.rasterize(
                shapes=shapes, out_shape=tile_geobox.shape, transform=tile_geobox.transform
            )

            # Convert numpy array to a full xarray.DataArray
            tile_raster_ds = wrap_xr(im=tile_raster_np, gbox=tile_geobox)

            # Write the raster to disk.
            cog_bytes = to_cog(tile_raster_ds)
            if check_if_s3_uri(tile_raster_fp):
                save_blob_to_s3(data=cog_bytes, url=tile_raster_fp).compute()
            else:
                save_blob_to_file(data=cog_bytes, url=tile_raster_fp).compute()
            _log.info(f"Exported raster data to {tile_raster_fp}")
        else:
            _log.info(f"{tile_raster_fp} already exists, skipping...")
