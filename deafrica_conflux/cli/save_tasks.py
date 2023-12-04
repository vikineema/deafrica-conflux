import logging
import os
import queue
from threading import Thread

import click
import datacube
import fsspec
from odc.dscache import create_cache
from odc.dscache.apps.slurpy import EOS, qmap
from odc.dscache.tools import (
    bin_dataset_stream,
    dataset_count,
    db_connect,
    dictionary_from_product_list,
    mk_raw2ds,
    ordered_dss,
    raw_dataset_stream,
)
from odc.dscache.tools.tiling import parse_gridspec_with_name
from odc.stats.model import DateTimeRange
from tqdm import tqdm

from deafrica_conflux.cli.logs import logging_setup
from deafrica_conflux.hopper import bin_solar_day, persist
from deafrica_conflux.io import (
    check_dir_exists,
    check_file_exists,
    check_if_s3_uri,
    find_geotiff_files,
)
from deafrica_conflux.text import parse_tile_ids


@click.command("save-tasks", help="Prepare tasks for drill.", no_args_is_help=True)
@click.option("-v", "--verbose", default=1, count=True)
@click.option("--grid-name", type=str, help="Grid name africa_{10|20|30|60}", default="africa_30")
@click.option(
    "--product",
    type=str,
    help="Datacube product to search datasets for.",
    default="wofs_ls",
)
@click.option(
    "--temporal-range",
    type=str,
    help=(
        "Only extract datasets for a given time range," "Example '2020-05--P1M' month of May 2020"
    ),
)
@click.option(
    "--complevel",
    type=int,
    default=6,
    help="Compression setting for zstandard 1-fast, 9+ good but slow",
)
@click.option(
    "--polygons-rasters-directory",
    type=str,
    help="Path to the directory containing the polygons raster files.",
)
@click.option(
    "--pattern",
    default=".*",  # noqa W605
    help="Regular expression for filename matching when searching for the polygons raster files.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=True,
    help="Overwrite existing cache file.",
)
@click.option("--output-directory", type=str, help="Directory to write the cache file to.")
def save_tasks(
    verbose,
    grid_name,
    product,
    temporal_range,
    complevel,
    polygons_rasters_directory,
    pattern,
    overwrite,
    output_directory,
):
    # Set up logger.
    logging_setup(verbose)
    _log = logging.getLogger(__name__)

    # Support pathlib Paths.
    polygons_rasters_directory = str(polygons_rasters_directory)
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

    # Validate the product
    products = [product]
    # Connect to the datacube.
    dc = datacube.Datacube()
    # Get all products.
    all_products = {p.name: p for p in dc.index.products.get_all()}
    if len(products) == 0:
        raise ValueError("Have to supply at least one product")
    else:
        for p in products:
            if p not in all_products:
                raise ValueError(f"No such product found: {p}")

    # Parse the temporal range.
    temporal_range_ = DateTimeRange(temporal_range)

    output_db_fn = f"{product}_{temporal_range_.short}.db"
    output_db_fp = os.path.join(output_directory, output_db_fn)

    # Check if the output file exists.
    if check_file_exists(output_db_fp):
        if overwrite:
            fs.delete(output_db_fp, recursive=True)
            _log.info(f"Deleted {output_db_fp}")
            # Delete the local file created before uploading to s3.
            if is_s3:
                if check_file_exists(output_db_fn):
                    fsspec.filesystem("file").delete(output_db_fn)
                    _log.info(f"Deleted local file created before uploading to s3 {output_db_fn}")
        else:
            raise FileExistsError(f"{output_db_fp} exists!")

    # Create the query to find the datasets.
    query = {"time": (temporal_range_.start, temporal_range_.end)}
    _log.info(f"Query: {query}")

    _log.info("Getting dataset counts")
    counts = {p: dataset_count(dc.index, product=p, **query) for p in products}

    n_total = 0
    for p, c in counts.items():
        _log.info(f"..{p}: {c:8,d}")
        n_total += c

    if n_total == 0:
        raise ValueError("No datasets found")

    _log.info("Training compression dictionary...")
    zdict = dictionary_from_product_list(dc, products, samples_per_product=50, query=query)
    _log.info("Done")

    if is_s3:
        cache = create_cache(output_db_fn, zdict=zdict, complevel=complevel, truncate=True)
    else:
        cache = create_cache(output_db_fp, zdict=zdict, complevel=complevel, truncate=True)

    raw2ds = mk_raw2ds(all_products)

    def db_task(products, conn, q):
        for p in products:
            if len(query) == 0:
                dss = map(raw2ds, raw_dataset_stream(p, conn))
            else:
                dss = ordered_dss(dc, product=p, **query)

            for ds in dss:
                q.put(ds)
        q.put(EOS)

    conn = db_connect()
    q = queue.Queue(maxsize=10_000)
    db_thread = Thread(target=db_task, args=(products, conn, q))
    db_thread.start()

    dss = qmap(lambda ds: ds, q, eos_marker=EOS)
    dss = cache.tee(dss)

    cells = {}
    grid, gridspec = parse_gridspec_with_name(grid_name)
    cache.add_grid(gridspec, grid)

    cfg = dict(grid=grid)
    cache.append_info_dict("stats/", dict(config=cfg))

    dss = bin_dataset_stream(gridspec, dss, cells, persist=persist)

    label = f"Processing {n_total:8,d} {product} datasets"
    with tqdm(dss, desc=label, total=n_total) as dss:
        for _ in dss:
            pass

    # Find the required tiles.
    _log.info(f"Total bins: {len(cells):d}")
    _log.info("Filtering bins by required tiles...")
    geotiff_files = find_geotiff_files(path=polygons_rasters_directory, pattern=pattern)

    tiles_ids = [parse_tile_ids(file) for file in geotiff_files]
    _log.info(f"Found {len(tiles_ids)} tiles.")
    _log.debug(f"Tile ids: {tiles_ids}")

    # Filter cells by tile ids.
    cells = {k: v for k, v in cells.items() if k in tiles_ids}
    _log.info(f"Total bins: {len(cells):d}")

    tasks = bin_solar_day(cells)

    # Remove duplicate source uuids.
    # Duplicates occur when queried datasets are captured around UTC midnight
    # and around weekly boundary
    tasks = {k: set(dss) for k, dss in tasks.items()}

    tasks_uuid = {k: [ds.id for ds in dss] for k, dss in tasks.items()}

    all_ids = set()
    for k, dss in tasks_uuid.items():
        all_ids.update(dss)
    _log.info(f"Total of {len(all_ids):,d} unique dataset IDs after filtering")

    label = f"Saving {len(tasks)} tasks to disk"
    with tqdm(tasks_uuid.items(), desc=label, total=len(tasks_uuid)) as groups:
        for group in groups:
            cache.add_grid_tile(grid, group[0], group[1])

    db_thread.join()
    cache.close()

    if is_s3:
        fs.upload(output_db_fn, output_db_fp, recursive=False)
        fsspec.filesystem("file").delete(output_db_fn)

    _log.info(f"Cache file written to {output_db_fp}")

    # pylint:disable=too-many-locals
    csv_path = os.path.join(output_directory, f"{product}_{temporal_range_.short}.csv")
    _log.info(f"Writing summary to {csv_path}")
    with fs.open(csv_path, "wt", encoding="utf8") as f:
        f.write('"T","X","Y","datasets","days"\n')
        for p, x, y in sorted(tasks):
            dss = tasks[(p, x, y)]
            n_dss = len(dss)
            n_days = len(set(ds.time.date() for ds in dss))
            line = f'"{p}", {x:+05d}, {y:+05d}, {n_dss:4d}, {n_days:4d}\n'
            f.write(line)
