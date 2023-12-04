import click

import deafrica_conflux.__version__
from deafrica_conflux.cli.db_to_csv import db_to_csv
from deafrica_conflux.cli.filter_from_queue import filter_from_queue
from deafrica_conflux.cli.publish_tasks import publish_tasks
from deafrica_conflux.cli.rasterise_polygons import rasterise_polygons
from deafrica_conflux.cli.run_from_queue import run_from_sqs_queue
from deafrica_conflux.cli.run_from_txt import run_from_txt
from deafrica_conflux.cli.save_tasks import save_tasks
from deafrica_conflux.cli.stack import stack


@click.version_option(package_name="deafrica_conflux", version=deafrica_conflux.__version__)
@click.group(help="Run deafrica-conflux.")
def main():
    pass


main.add_command(rasterise_polygons)
main.add_command(save_tasks)
main.add_command(run_from_txt)
main.add_command(publish_tasks)
main.add_command(run_from_sqs_queue)
main.add_command(stack)
main.add_command(db_to_csv)
main.add_command(filter_from_queue)
