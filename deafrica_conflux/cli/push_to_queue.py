import logging

import boto3
import click

import deafrica_conflux.queues
from deafrica_conflux.cli.logs import logging_setup


@click.command("push-to-sqs-queue", no_args_is_help=True)
@click.option(
    "--text-file-path",
    type=click.Path(),
    required=True,
    help="REQUIRED. Path to text file to push to queue.",
)
@click.option("--queue-name", required=True, help="REQUIRED. Queue name to push to.")
@click.option(
    "--max-retries",
    default=10,
    help="Maximum number of times to retry sending/receiving messages to/from a SQS queue.",
)
@click.option("-v", "--verbose", count=True)
def push_to_sqs_queue(text_file_path, queue_name, max_retries, verbose):
    """
    Push dataset ids from the lines of a text file to a SQS queue.
    """
    # Cribbed from datacube-alchemist
    logging_setup(verbose)
    _log = logging.getLogger(__name__)  # noqa F841

    # Create an sqs client.
    sqs_client = boto3.client("sqs")

    failed_to_push = deafrica_conflux.queues.push_dataset_ids_to_queue_from_txt(
        text_file_path=text_file_path,
        queue_name=queue_name,
        max_retries=max_retries,
        sqs_client=sqs_client,
    )

    if failed_to_push:
        # Push the failed dataset ids to the deadletter queue.
        deadletter_queue_name = f"{queue_name}-deadletter"
        deadletter_queue_url = deafrica_conflux.queues.get_queue_url(
            queue_name=deadletter_queue_name, sqs_client=sqs_client
        )

        for idx in failed_to_push:
            deafrica_conflux.queues.move_to_deadletter_queue(
                deadletter_queue_name=deadletter_queue_url,
                message_body=idx,
                max_retries=max_retries,
                sqs_client=sqs_client,
            )
