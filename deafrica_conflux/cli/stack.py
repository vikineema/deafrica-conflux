import json
import logging

import click
import fsspec

from deafrica_conflux.cli.logs import logging_setup
from deafrica_conflux.io import (
    check_dir_exists,
    check_file_exists,
    check_if_s3_uri,
    find_parquet_files,
)
from deafrica_conflux.stack import stack_waterbodies_parquet_to_csv


@click.command("stack", no_args_is_help=True)
@click.option("-v", "--verbose", default=1, count=True)
@click.option(
    "--drill-output-directory",
    type=str,
    # Don't mandate existence since this might be s3://.
    help="Path to the directory containing the parquet files output during polygon drill.",
)
@click.option(
    "--pattern",
    default=".*",  # noqa W605
    help="Regular expression for filename matching.",
)
@click.option(
    "--output-directory",
    type=str,
    help="Output directory for waterbodies-style stack",
)
@click.option(
    "--polygons-ids-mapping-file",
    type=str,
    help="JSON file mapping numerical polygons ids (WB_ID) to string polygons ids (UID).",
)
def stack(verbose, drill_output_directory, pattern, output_directory, polygons_ids_mapping_file):
    """
    Stack outputs of deafrica-conflux into csv formats.
    """
    # Set up logger.
    logging_setup(verbose)
    _log = logging.getLogger(__name__)

    # Support pathlib Paths
    output_directory = str(output_directory)
    drill_output_directory = str(drill_output_directory)

    if not check_dir_exists(drill_output_directory):
        _log.error(f"Directory {drill_output_directory} does not exist!")
        raise FileNotFoundError(f"Directory {drill_output_directory} does not exist!)")

    # Create the output directory if it does not exist.
    if not check_dir_exists(output_directory):
        if check_if_s3_uri(output_directory):
            fsspec.filesystem("s3").makedirs(output_directory, exist_ok=True)
        else:
            fsspec.filesystem("file").makedirs(output_directory, exist_ok=True)
        _log.info(f"Created directory {output_directory}")

    if not check_file_exists(polygons_ids_mapping_file):
        _log.error(f"File {polygons_ids_mapping_file} does not exist!")
        raise FileNotFoundError(f"File {polygons_ids_mapping_file} does not exist!)")

    # Read the polygons ids mapping file.
    if check_if_s3_uri(polygons_ids_mapping_file):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    with fs.open(polygons_ids_mapping_file) as f:
        polygons_ids_mapping = json.load(f)

    # Find all the drill output parquet files
    drill_output_pq_files = find_parquet_files(path=drill_output_directory, pattern=".*")

    stack_waterbodies_parquet_to_csv(
        parquet_file_paths=drill_output_pq_files,
        output_directory=output_directory,
        polygons_ids_mapping=polygons_ids_mapping,
    )
