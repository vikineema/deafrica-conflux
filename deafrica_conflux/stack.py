"""Stack Parquet scene outputs into other formats.

Lots of this code is domain-specific and not intended to be fully general.

Matthew Alger
Geoscience Australia
2021
"""

import collections
import concurrent.futures
import enum
import logging
import os
from pathlib import Path

import fsspec
import geohash
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, scoped_session, sessionmaker
from tqdm.auto import tqdm

from deafrica_conflux.db import (
    Engine,
    Waterbody,
    WaterbodyObservation,
    create_waterbody_tables,
    drop_waterbody_tables,
    get_engine_waterbodies,
    get_or_create,
)
from deafrica_conflux.io import (
    check_dir_exists,
    check_if_s3_uri,
    find_parquet_files,
    load_parquet_file,
)

_log = logging.getLogger(__name__)


class StackMode(enum.Enum):
    WATERBODIES = "waterbodies"
    WATERBODIES_DB = "waterbodies_db"


def update_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    """
    Update the data in the timeseries DataFrame.

    Arguments
    ---------
    df : pd.DataFrame
        The polygon base timeseries result.

    Returns
    -------
    pd.DataFrame
        The polygon base timeseries.
    """
    assert "date" in df.columns

    keep = np.unique(df["date"])

    # Group by day
    df = df.groupby(pd.Grouper(key="date", axis=0, freq="D")).sum()
    # Remove filler rows.
    df = df[df.index.isin(keep)]

    df["pc_wet"] = (df["px_wet"] / df["px_total"]) * 100.0
    df["pc_dry"] = (df["px_dry"] / df["px_total"]) * 100.0
    df["pc_invalid"] = (df["px_invalid"] / df["px_total"]) * 100.0

    # If the proportion of the waterbody missing is greater than 10%,
    # set the values for pc_wet and pc_dry to nan.
    df.loc[df["pc_invalid"] > 10.0, "pc_wet"] = np.nan
    df.loc[df["pc_invalid"] > 10.0, "pc_dry"] = np.nan

    df.sort_index(inplace=True)
    return df


def stack_waterbodies_parquet_to_csv(
    parquet_file_paths: [str | Path],
    output_directory: str | Path,
    polygons_ids_mapping: dict[str, str],
):
    """
    Stack Parquet files into CSVs.

    Arguments
    ---------
    parquet_file_paths : [str | Path]
        List of paths to Parquet files to stack.

    output_directory : str | Path
        Path to output directory.

    polygons_ids_mapping: bool
        Dictionary mapping numerical polygon ids (WB_ID) to string polygon ids (UID)
    """
    # "Support" pathlib Paths.
    parquet_file_paths = [str(pq_file_path) for pq_file_path in parquet_file_paths]
    output_directory = str(output_directory)

    assert polygons_ids_mapping

    # id -> [series of date x bands]
    id_to_series = collections.defaultdict(list)
    n_total = len(parquet_file_paths)
    label = f"Reading {n_total} drill output parquet files."
    with tqdm(parquet_file_paths, desc=label, total=n_total) as parquet_file_paths:
        for pq_file_path in parquet_file_paths:
            df = load_parquet_file(pq_file_path)
            # df is ids x bands
            # for each ID...
            for uid, series in df.iterrows():
                series.name = series.date
                uid = polygons_ids_mapping[str(uid)]
                id_to_series[uid].append(series)

    n_total = len(id_to_series.items())
    for i, item in enumerate(id_to_series.items()):
        uid, seriess = item
        _log.info(f"Writing csv file for polygon {uid} ({i + 1}/{n_total})")

        df = pd.DataFrame(seriess)
        df = update_timeseries(df=df)

        output_file_parent_directory = os.path.join(output_directory, f"{uid[:4]}")
        output_file_path = os.path.join(output_file_parent_directory, f"{uid}.csv")

        if check_if_s3_uri(output_file_parent_directory):
            fs = fsspec.filesystem("s3")
        else:
            fs = fsspec.filesystem("file")

        if not check_dir_exists(output_file_parent_directory):
            fs.mkdirs(output_file_parent_directory, exist_ok=True)
            _log.info(f"Created directory: {output_file_parent_directory}")

        with fs.open(output_file_path, "w") as f:
            df.to_csv(f, index_label="date")

        _log.info(f"CSV file written to {output_file_path}")


def get_waterbody_key(uid: str, session: Session):
    """
    Create or get a unique key from the database.
    """
    # decode into a coordinate
    # uid format is gh_version
    gh = uid.split("_")[0]
    lat, lon = geohash.decode(gh)
    defaults = {
        "geofabric_name": "",
        "centroid_lat": lat,
        "centroid_lon": lon,
    }
    inst, _ = get_or_create(session, Waterbody, wb_name=uid, defaults=defaults)
    return inst.wb_id


def stack_waterbodies_parquet_to_db(
    parquet_file_paths: [str | Path],
    verbose: bool = False,
    engine: Engine = None,
    uids: {str} = None,
    drop: bool = False,
):
    """
    Stack Parquet files into the waterbodies interstitial DB.

    Arguments
    ---------
    parquet_file_paths : [str | Path]
        List of paths to Parquet files to stack.

    verbose : bool

    engine: sqlalchemy.engine.Engine
        Database engine. Default postgres, which is
        connected to if engine=None.

    uids : {uids}
        Set of waterbody IDs. If not specified, guessed from
        parquet files, but that's slower.

    drop : bool
        Whether to drop the database. Default False.
    """
    parquet_file_paths = [str(pq_file_path) for pq_file_path in parquet_file_paths]

    if verbose:
        parquet_file_paths = tqdm(parquet_file_paths)

    # connect to the db
    if not engine:
        engine = get_engine_waterbodies()

    Session = sessionmaker(bind=engine)
    session = Session()

    # drop tables if requested
    if drop:
        drop_waterbody_tables(engine)

    # ensure tables exist
    create_waterbody_tables(engine)

    if not uids:
        uids = set()

    # confirm all the UIDs exist in the db
    uid_to_key = {}
    uids_ = uids
    if verbose:
        uids_ = tqdm(uids)
    for uid in uids_:
        key = get_waterbody_key(uid, session)
        uid_to_key[uid] = key

    for pq_file_path in parquet_file_paths:
        # read the table in...
        df = load_parquet_file(pq_file_path)
        # df is ids x bands
        # for each ID...
        obss = []
        for uid, series in df.iterrows():
            if uid not in uid_to_key:
                # add this uid
                key = get_waterbody_key(uid, session)
                uid_to_key[uid] = key

            key = uid_to_key[uid]
            obs = WaterbodyObservation(
                wb_id=key,
                wet_pixel_count=series.wet_pixel_count,
                wet_percentage=series.wet_percentage,
                invalid_percentage=series.invalid_percentage,
                platform="UNK",
                date=series.date,
            )
            obss.append(obs)
        # basically just hoping that these don't exist already
        # TODO: Insert or update
        session.bulk_save_objects(obss)
        session.commit()


def stack_waterbodies_db_to_csv(
    output_directory: str | Path,
    verbose: bool = False,
    uids: {str} = None,
    remove_duplicated_data: bool = True,
    engine=None,
    n_workers: int = 8,
    index_num: int = 0,
    split_num: int = 1,
):
    """
    Write waterbodies CSVs out from the interstitial DB.

    Arguments
    ---------
    output_directory : str | Path
        Path to write CSVs to.

    verbose : bool

    engine: sqlalchemy.engine.Engine
        Database engine. Default postgres, which is
        connected to if engine=None.

    uids : {uids}
        Set of waterbody IDs. If not specified, use all.

    remove_duplicated_data: bool
        Remove timeseries duplicated data or not

    engine : Engine
        Database engine. If not specified, use the
        Waterbodies engine.

    n_workers : int
        Number of threads to connect to the database with.

    index_num: int
        Index number of waterbodies ID list. Use to create the subset of
        waterbodies, then generate relative CSV files.

    split_num: int
        Number of chunks after split overall waterbodies ID list

    """
    # "Support" pathlib Paths.
    output_directory = str(output_directory)

    # connect to the db
    if not engine:
        engine = get_engine_waterbodies()

    session_factory = sessionmaker(bind=engine)
    Session = scoped_session(session_factory)

    # Iterate over waterbodies.

    def thread_run(wb: Waterbody):
        session = Session()

        # get all observations
        _log.debug(f"Processing {wb.wb_name}")
        obs = (
            session.query(WaterbodyObservation)
            .filter(WaterbodyObservation.wb_id == wb.wb_id)
            .order_by(WaterbodyObservation.date.asc())
            .all()
        )

        rows = [
            {
                "date": ob.date,
                "wet_percentage": ob.wet_percentage,
                "wet_pixel_count": ob.wet_pixel_count,
                "invalid_percentage": ob.invalid_percentage,
            }
            for ob in obs
        ]

        df = pd.DataFrame(
            rows, columns=["date", "wet_percentage", "wet_pixel_count", "invalid_percentage"]
        )
        if remove_duplicated_data:
            df = remove_duplicated_data(df)

        # The pc_missing should not in final WaterBodies result
        df.drop(columns=["pc_missing"], inplace=True)

        output_file_name = os.path.join(output_directory, wb.wb_name[:4], wb.wb_name + ".csv")

        if check_if_s3_uri(output_file_name):
            fs = fsspec.filesystem("s3")
        else:
            fs = fsspec.filesystem("file")

        with fs.open(output_file_name, "w") as f:
            df.to_csv(f, header=True, index=False)

        Session.remove()

    session = Session()
    if not uids:
        # query all
        waterbodies = session.query(Waterbody).all()
    else:
        # query some
        waterbodies = session.query(Waterbody).filter(Waterbody.wb_name.in_(uids)).all()

    # generate the waterbodies list
    waterbodies = np.array_split(waterbodies, split_num)[index_num]

    # Write all CSVs with a thread pool.
    with tqdm(total=len(waterbodies)) as bar:
        # https://stackoverflow.com/a/63834834/1105803
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(thread_run, wb): wb for wb in waterbodies}
            for future in concurrent.futures.as_completed(futures):
                # _ = futures[future]
                bar.update(1)

    Session.remove()


def stack_parquet(
    path: str | Path,
    pattern: str = ".*",
    mode: StackMode = StackMode.WATERBODIES,
    verbose: bool = False,
    **kwargs,
):
    """
    Stack Parquet files.

    Arguments
    ---------
    path : str | Path
        Path (s3 or local) to search for parquet files.

    pattern : str
        Regex to match file names against.

    mode : StackMode
        Method of stacking. Default is like DE Africa Waterbodies v1,
        a collection of polygon CSVs.

    verbose : bool

    **kwargs
        Passed to underlying stack method.
    """
    # "Support" pathlib Paths.
    path = str(path)

    _log.info(f"Begin to query {path} with pattern {pattern}")

    paths = find_parquet_files(path, pattern)

    if mode == StackMode.WATERBODIES:
        return stack_waterbodies_parquet_to_csv(parquet_file_paths=paths, verbose=verbose, **kwargs)
    if mode == StackMode.WATERBODIES_DB:
        return stack_waterbodies_parquet_to_db(parquet_file_paths=paths, verbose=verbose, **kwargs)
