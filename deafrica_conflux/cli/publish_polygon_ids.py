import json
import logging
import time

import boto3
import click
import fsspec

from deafrica_conflux.cli.logs import logging_setup
from deafrica_conflux.io import check_file_exists, check_if_s3_uri
from deafrica_conflux.queues import get_queue_url, send_batch_with_retry


@click.command(
    "publish-polygon-ids",
    no_args_is_help=True,
)
@click.option("-v", "--verbose", default=1, count=True)
@click.option(
    "--ids-sqs-queue",
    type=str,
    help="SQS Queue to publish the polygons ids to.",
)
@click.option(
    "--polygons-ids-mapping-file",
    type=str,
    help="JSON file mapping numerical polygons ids (WB_ID) to string polygons ids (UID).",
)
def publish_polygon_ids(
    verbose,
    ids_sqs_queue,
    polygon_ids_mapping_file,
):
    """Publish polygon ids to SQS queue."""
    # Set up logger.
    logging_setup(verbose)
    _log = logging.getLogger(__name__)

    # Support pathlib paths.
    polygon_ids_mapping_file = str(polygon_ids_mapping_file)

    if not check_file_exists(polygon_ids_mapping_file):
        _log.error(f"File {polygon_ids_mapping_file} does not exist!")
        raise FileNotFoundError(f"File {polygon_ids_mapping_file} does not exist!)")

    # Read the polygons ids mapping file.
    if check_if_s3_uri(polygon_ids_mapping_file):
        fs = fsspec.filesystem("s3")
    else:
        fs = fsspec.filesystem("file")

    with fs.open(polygon_ids_mapping_file) as f:
        polygon_ids_mapping = json.load(f)

    # Find all the drill output parquet files
    polygon_ids = list(polygon_ids_mapping.values())

    sqs_client = boto3.client("sqs")
    ids_sqs_queue_url = get_queue_url(queue_name=ids_sqs_queue, sqs_client=sqs_client)

    # Check if there are any messages in the queue.
    # If there are any messages purge the queue.
    response = sqs_client.get_queue_attributes(QueueUrl=ids_sqs_queue_url, AttributeNames=["All"])
    if float(response["Attributes"]["ApproximateNumberOfMessages"]) > 0:
        _log.info(f"Purging queue {ids_sqs_queue_url}...")
        response = sqs_client.purge_queue(QueueUrl=ids_sqs_queue_url)
        time.sleep(60)  # Delay for 1 minute
        _log.info(f"Purge of queue {ids_sqs_queue_url} is complete.")

    _, failed_to_push = send_batch_with_retry(
        queue_url=ids_sqs_queue_url, messages=polygon_ids, max_retries=10, sqs_client=sqs_client
    )
    if failed_to_push:
        _log.error(f"Failed to push the polygon ids: {failed_to_push}")
