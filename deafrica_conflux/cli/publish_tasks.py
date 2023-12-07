import logging
import time

import boto3
import click
import fsspec
from odc import dscache
from odc.aws import s3_download
from odc.stats._cli_common import parse_all_tasks

from deafrica_conflux.cli.common import MutuallyExclusiveOption
from deafrica_conflux.cli.logs import logging_setup
from deafrica_conflux.io import check_file_exists, check_if_s3_uri
from deafrica_conflux.queues import get_queue_url, purge_queue, send_batch_with_retry
from deafrica_conflux.text import task_id_to_string


@click.command(
    "publish-tasks",
    no_args_is_help=True,
)
@click.option("-v", "--verbose", default=1, count=True)
@click.option("--cachedb-file-path", type=str, help="File path to the cache file database.")
@click.option(
    "--tasks-sqs-queue",
    type=str,
    help="SQS Queue to publish the tasks ids to.",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["tasks_text_file"],
)
@click.option(
    "--tasks-text-file",
    type=str,
    help="Text file to write the tasks ids to.",
    cls=MutuallyExclusiveOption,
    mutually_exclusive=["tasks_sqs_queue"],
)
@click.option("--task-filter", type=str, default="")
def publish_tasks(
    verbose,
    cachedb_file_path,
    tasks_sqs_queue,
    tasks_text_file,
    task_filter,
):
    """
        Publish tasks to SQS queue or text file.

    \b
    Task filter can be one of the 3 things
    1. Comma-separated triplet: period,x,y or 'x[+-]<int>/y[+-]<int>/period
       2019--P1Y,+003,-004
       2019--P1Y/3/-4          `/` is also accepted
       x+003/y-004/2019--P1Y   is accepted as well
    2. A zero based index
    3. A slice following python convention <start>:<stop>[:<step]
        ::10 -- every tenth task: 0,10,20,..
       1::10 -- every tenth but skip first one 1, 11, 21 ..
        :100 -- first 100 tasks

    If no tasks are supplied all tasks will be published the queue or text file.
    """
    # Set up logger.
    logging_setup(verbose)
    _log = logging.getLogger(__name__)

    # Verify
    if (tasks_sqs_queue and tasks_text_file) or (not tasks_sqs_queue and not tasks_text_file):
        raise ValueError("Provide EITHER tasks_sqs_queue OR tasks_text_file!")

    # Support pathlib Paths.
    cachedb_file_path = str(cachedb_file_path)
    if tasks_text_file is not None:
        tasks_text_file = str(tasks_text_file)

    # Check if the cache db file exists.
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

    # Read the cache file
    cache = dscache.open_ro(cachedb_file_path)

    # Get all the tiles in the file db.
    cfg = cache.get_info_dict("stats/config")
    grid = cfg["grid"]

    all_tasks = sorted(idx for idx, _ in cache.tiles(grid)) if cache else []
    _log.info(f"Found {len(all_tasks):,d} tasks in the file")

    # Filter the tasks using the task filter.
    if len(task_filter) == 0:
        tasks = all_tasks
        _log.info(f"Found {len(all_tasks):,d} tasks.")
    else:
        tasks = parse_all_tasks(task_filter, all_tasks)
        _log.info(f"Found {len(tasks):,d} tasks after filtering using filter {task_filter}")

    tasks_str = [task_id_to_string(tidx) for tidx in tasks]

    if tasks_sqs_queue:
        sqs_client = boto3.client("sqs")
        tasks_sqs_queue_url = get_queue_url(queue_name=tasks_sqs_queue, sqs_client=sqs_client)

        # Check if there are any messages in the queue.
        # If there are any messages purge the queue.
        purge_queue(queue_url=tasks_sqs_queue_url, sqs_client=sqs_client)

        _, failed_to_push = send_batch_with_retry(
            queue_url=tasks_sqs_queue_url, messages=tasks_str, max_retries=10, sqs_client=sqs_client
        )
        if failed_to_push:
            _log.error(f"Failed to push the tasks: {failed_to_push}")
    elif tasks_text_file:
        if check_if_s3_uri(tasks_text_file):
            fs = fsspec.filesystem("s3")
        else:
            fs = fsspec.filesystem("file")
        with fs.open(tasks_text_file, "w") as file:
            for task in tasks_str:
                file.write(f"{task}\n")
        _log.info(f"{len(tasks_str)} tasks written to: {tasks_text_file}.")
