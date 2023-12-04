import logging
import os
from importlib import import_module

import click
import datacube
import fsspec
from odc import dscache
from odc.aws import s3_download
from rasterio.errors import RasterioIOError

from deafrica_conflux.cli.logs import logging_setup
from deafrica_conflux.db import get_engine_waterbodies
from deafrica_conflux.drill import drill
from deafrica_conflux.io import (
    check_file_exists,
    check_if_s3_uri,
    table_exists,
    write_table_to_parquet,
)
from deafrica_conflux.plugins.utils import run_plugin, validate_plugin
from deafrica_conflux.stack import stack_waterbodies_parquet_to_db


@click.command(
    "run-from-txt",
    no_args_is_help=True,
    help="Run deafrica-conflux on tasks from a text file.",
)
@click.option("-v", "--verbose", default=1, count=True)
@click.option("--cachedb-file-path", type=str, help="File path to the cache file database.")
@click.option(
    "--tasks-text-file",
    type=str,
    help="Text file to get tasks ids from.",
)
@click.option(
    "--plugin-name",
    type=str,
    help="Name of the plugin. Plugin file must be in the \
        deafrica_conflux/plugins/ directory.",
)
@click.option(
    "--polygons-rasters-directory",
    type=str,
    help="Path to the directory containing the polygons raster files.",
)
@click.option("--output-directory", type=str, help="Directory to write the drill outputs to.")
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun tasks that have already been processed.",
)
@click.option("--db/--no-db", default=False, help="Write to the Waterbodies database.")
@click.option(
    "--dump-empty-dataframe/--not-dump-empty-dataframe",
    default=False,
    help="Not matter DataFrame is empty or not, always as it as Parquet file.",
)
def run_from_txt(
    verbose,
    cachedb_file_path,
    tasks_text_file,
    plugin_name,
    polygons_rasters_directory,
    output_directory,
    overwrite,
    db,
    dump_empty_dataframe,
):
    # Set up logger.
    logging_setup(verbose)
    _log = logging.getLogger(__name__)

    # Support pathlib Paths
    cachedb_file_path = str(cachedb_file_path)
    tasks_text_file = str(tasks_text_file)
    polygons_rasters_directory = str(polygons_rasters_directory)
    output_directory = str(output_directory)

    # Read the plugin as a Python module.
    module = import_module(f"deafrica_conflux.plugins.{plugin_name}")
    plugin_file = module.__file__
    plugin = run_plugin(plugin_file)
    _log.info(f"Using plugin {plugin_file}")
    validate_plugin(plugin)

    # Get the drill name from the plugin
    drill_name = plugin.product_name

    if not check_file_exists(cachedb_file_path):
        _log.error(f"Could not find the database file {cachedb_file_path}!")
        raise FileNotFoundError(f"{cachedb_file_path} does not exist!")
    else:
        if check_if_s3_uri(cachedb_file_path):
            cachedb_file_path = s3_download(cachedb_file_path)
            if not check_file_exists(cachedb_file_path):
                _log.error(f"{cachedb_file_path} did not download!")
                raise FileNotFoundError(
                    f"{cachedb_file_path} does not exist! File did not download."
                )

    if not check_file_exists(tasks_text_file):
        _log.error(f"Could not find the text file {tasks_text_file}!")
        raise FileNotFoundError(f"Could not find text file {tasks_text_file}!")

    # Read task ids from the S3 URI or File URI.
    if check_if_s3_uri(tasks_text_file):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    with fs.open(tasks_text_file, "r") as file:
        tasks = [line.strip() for line in file]
    _log.info(f"Read {len(tasks)} tasks from file.")
    _log.debug(f"Read {tasks} from file.")

    if db:
        engine = get_engine_waterbodies()

    # Connect to the datacube
    dc = datacube.Datacube(app="deafrica-conflux-drill")

    # Read the cache file
    cache = dscache.open_ro(cachedb_file_path)

    failed_tasks = []
    for i, task in enumerate(tasks):
        _log.info(f"Processing {task} ({i + 1}/{len(tasks)})")

        # Get the tasks output file name.
        if not overwrite:
            _log.info(f"Checking existence of {task}")
            exists = table_exists(
                drill_name=drill_name, task_id_string=task, output_directory=output_directory
            )
        if overwrite or not exists:
            try:
                # Perform the polygon drill.
                table = drill(
                    plugin=plugin,
                    task_id_string=task,
                    cache=cache,
                    polygon_rasters_split_by_tile_directory=polygons_rasters_directory,
                    dc=dc,
                )
                # Write the table to a parquet file.
                if (dump_empty_dataframe) or (not table.empty):
                    pq_file_name = write_table_to_parquet(
                        drill_name=drill_name,
                        task_id_string=task,
                        table=table,
                        output_directory=output_directory,
                    )
                    if db:
                        _log.info(f"Writing {pq_file_name} to database")
                        stack_waterbodies_parquet_to_db(
                            parquet_file_paths=[pq_file_name],
                            verbose=verbose,
                            engine=engine,
                            drop=False,
                        )

            except KeyError as keyerr:
                _log.exception(f"Found {task} has KeyError: {str(keyerr)}")
                failed_tasks = [].append(task)
            except TypeError as typeerr:
                _log.exception(f"Found {task} has TypeError: {str(typeerr)}")
                failed_tasks.append(task)
            except RasterioIOError as ioerror:
                _log.exception(f"Found {task} has RasterioIOError: {str(ioerror)}")
                failed_tasks.append(task)
            except ValueError as valueerror:
                _log.exception(f"Found {task} has ValueError: {str(valueerror)}")
                failed_tasks.append(task)
            else:
                _log.info(f"{task} successful")
        else:
            _log.info(f"{task} already exists, skipping")

        if failed_tasks:
            # Write the failed dataset ids to a text file.
            parent_folder, file_name = os.path.split(tasks_text_file)
            file, file_extension = os.path.splitext(file_name)
            failed_tasks_text_file = os.path.join(
                parent_folder, file + "_failed_tasks" + file_extension
            )

            with fs.open(failed_tasks_text_file, "a") as file:
                for task in failed_tasks:
                    file.write(f"{task}\n")

            _log.info(f"Failed tasks {failed_tasks} written to: {failed_tasks_text_file}.")
