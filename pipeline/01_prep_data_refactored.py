import os
import zipfile
import logging
from typing import Any, Dict, Optional

import duckdb
import numpy as np
import pandas as pd
import requests

DATA_DIR = "data"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-32m.zip")
DB_PATH = os.path.join(DATA_DIR, "movielens.db")
RANDOM_SEED = 4320


def configure_logging(log_path: str = "pipeline.log") -> logging.Logger:
    """Configure and return a module logger."""
    logger = logging.getLogger("prep_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Return a reproducible random number generator."""
    return np.random.default_rng(seed)


def ensure_data_dir(data_dir: str = DATA_DIR) -> None:
    """Create the data directory if it does not exist."""
    os.makedirs(data_dir, exist_ok=True)


def download_and_extract(
    logger: logging.Logger,
    zip_url: str = ZIP_URL,
    zip_path: str = ZIP_PATH,
    data_dir: str = DATA_DIR,
) -> None:
    """Download MovieLens and extract the CSV files into data_dir."""
    logger.info("Starting dataset download")

    with requests.get(zip_url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    logger.info("Download complete")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith(".csv"):
                filename = os.path.basename(member)
                out_path = os.path.join(data_dir, filename)
                with zip_ref.open(member) as source, open(out_path, "wb") as target:
                    target.write(source.read())

    logger.info("CSV extraction complete")


def validate_raw_files(logger: logging.Logger, data_dir: str = DATA_DIR) -> None:
    """Validate that all required raw MovieLens CSVs are present."""
    required = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]
    for filename in required:
        path = os.path.join(data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected file: {path}")
        logger.info("Validated raw file: %s", path)


def generate_synthetic_demographics(
    logger: logging.Logger,
    rng: np.random.Generator,
    data_dir: str = DATA_DIR,
) -> str:
    """
    Generate synthetic demographics for each user.

    The original MovieLens release does not contain demographics, so this function
    creates synthetic attributes needed for the project question.
    """
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    user_ids = np.sort(ratings["userId"].dropna().unique())

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    age_probs = [0.12, 0.18, 0.17, 0.16, 0.37]

    sex_values = ["Male", "Female"]
    sex_probs = [0.49, 0.51]

    users_demo = pd.DataFrame({
        "userId": user_ids,
        "age_group": rng.choice(age_groups, size=len(user_ids), p=age_probs),
        "sex": rng.choice(sex_values, size=len(user_ids), p=sex_probs),
        "synth_source": "ACS_sampled",
    })

    out_path = os.path.join(data_dir, "users_demo.csv")
    users_demo.to_csv(out_path, index=False)
    logger.info("Synthetic demographics created: %s rows", len(users_demo))
    return out_path


def create_adjusted_ratings(
    logger: logging.Logger,
    rng: np.random.Generator,
    data_dir: str = DATA_DIR,
) -> str:
    """
    Create adjusted ratings with a small injected demographic signal.

    This keeps behavior as the dominant predictor while ensuring demographics are
    not pure random noise.
    """
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_dir, "movies.csv"))
    users_demo = pd.read_csv(os.path.join(data_dir, "users_demo.csv"))

    df = ratings.merge(movies, on="movieId", how="inner")
    df = df.merge(users_demo, on="userId", how="inner")

    genre_preferences_by_age = {
        "18-24": {"Action": 0.15, "Comedy": 0.10, "Sci-Fi": 0.10},
        "25-34": {"Thriller": 0.10, "Drama": 0.10, "Romance": 0.05},
        "35-44": {"Drama": 0.15, "Crime": 0.10},
        "45-54": {"Documentary": 0.15, "Drama": 0.10},
        "55+": {"Documentary": 0.20, "Drama": 0.10, "War": 0.05},
    }

    genre_preferences_by_sex = {
        "Male": {"Action": 0.10, "Sci-Fi": 0.10, "War": 0.05},
        "Female": {"Romance": 0.10, "Drama": 0.10, "Children's": 0.05},
    }

    def adjust_rating(row: pd.Series) -> float:
        genres = row["genres"].split("|") if isinstance(row["genres"], str) else []

        age_bias = sum(
            genre_preferences_by_age.get(row["age_group"], {}).get(g, 0.0)
            for g in genres
        )
        sex_bias = sum(
            genre_preferences_by_sex.get(row["sex"], {}).get(g, 0.0)
            for g in genres
        )

        noise = rng.normal(0, 0.08)
        new_rating = row["rating"] + age_bias + sex_bias + noise
        return float(np.clip(new_rating, 0.5, 5.0))

    df["rating"] = df.apply(adjust_rating, axis=1)

    adjusted = df[["userId", "movieId", "rating", "timestamp"]].copy()
    out_path = os.path.join(data_dir, "ratings_adjusted.csv")
    adjusted.to_csv(out_path, index=False)
    logger.info("Adjusted ratings created: %s rows", len(adjusted))
    return out_path


def load_to_duckdb(
    logger: logging.Logger,
    db_path: str = DB_PATH,
    data_dir: str = DATA_DIR,
) -> duckdb.DuckDBPyConnection:
    """Load the prepared tables into DuckDB and return an open connection."""
    con = duckdb.connect(db_path)

    con.execute("""
        CREATE OR REPLACE TABLE ratings AS
        SELECT * FROM read_csv_auto('data/ratings_adjusted.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE movies AS
        SELECT * FROM read_csv_auto('data/movies.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE tags AS
        SELECT * FROM read_csv_auto('data/tags.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE links AS
        SELECT * FROM read_csv_auto('data/links.csv')
    """)

    con.execute("""
        CREATE OR REPLACE TABLE users_demo AS
        SELECT * FROM read_csv_auto('data/users_demo.csv')
    """)

    logger.info("Loaded all core tables into DuckDB")
    return con


def validate_database(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
) -> Dict[str, int]:
    """Run row-count and join validations against the DuckDB database."""
    tables = ["ratings", "movies", "tags", "links", "users_demo"]
    counts: Dict[str, int] = {}

    for table in tables:
        counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("%s row count: %s", table, counts[table])

    distinct_users = con.execute(
        "SELECT COUNT(DISTINCT userId) FROM ratings"
    ).fetchone()[0]

    users_demo_rows = con.execute(
        "SELECT COUNT(*) FROM users_demo"
    ).fetchone()[0]

    if distinct_users != users_demo_rows:
        raise ValueError(
            f"Mismatch: ratings has {distinct_users} distinct users but users_demo has {users_demo_rows} rows"
        )

    joined = con.execute("""
        SELECT COUNT(*)
        FROM ratings r
        JOIN users_demo u ON r.userId = u.userId
    """).fetchone()[0]

    ratings_count = counts["ratings"]

    if joined != ratings_count:
        raise ValueError(
            f"Join validation failed: ratings={ratings_count}, joined={joined}"
        )

    logger.info("Database validation passed")
    return counts


def export_parquet(
    con: duckdb.DuckDBPyConnection,
    logger: logging.Logger,
    data_dir: str = DATA_DIR,
) -> None:
    """Export the core tables to parquet for storage and sharing."""
    for table in ["ratings", "movies", "tags", "links", "users_demo"]:
        out_path = os.path.join(data_dir, f"{table}.parquet")
        con.execute(f"COPY {table} TO '{out_path}' (FORMAT PARQUET)")
        logger.info("Exported %s to parquet", table)


def cleanup_temp_files(
    logger: logging.Logger,
    data_dir: str = DATA_DIR,
    zip_path: str = ZIP_PATH,
) -> None:
    """Remove intermediate CSV and zip files after the database/parquet are built."""
    temp_files = [
        os.path.join(data_dir, "ratings.csv"),
        os.path.join(data_dir, "ratings_adjusted.csv"),
        os.path.join(data_dir, "movies.csv"),
        os.path.join(data_dir, "tags.csv"),
        os.path.join(data_dir, "links.csv"),
        os.path.join(data_dir, "users_demo.csv"),
        zip_path,
    ]

    for path in temp_files:
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info("Deleted temp file: %s", path)
        except Exception as exc:
            logger.warning("Could not delete %s: %s", path, exc)


def run_prep(
    cleanup: bool = False,
    export_parquet_files: bool = True,
    log_path: str = "pipeline.log",
) -> Dict[str, Any]:
    """
    Run the full preparation pipeline and return useful metadata.

    Set cleanup=False in notebooks so intermediate files remain available during
    interactive work. Set cleanup=True for a final batch run.
    """
    ensure_data_dir(DATA_DIR)
    logger = configure_logging(log_path)
    rng = get_rng()
    con: Optional[duckdb.DuckDBPyConnection] = None

    try:
        download_and_extract(logger)
        validate_raw_files(logger)
        generate_synthetic_demographics(logger, rng)
        create_adjusted_ratings(logger, rng)
        con = load_to_duckdb(logger)
        counts = validate_database(con, logger)

        if export_parquet_files:
            export_parquet(con, logger)

        return {
            "db_path": DB_PATH,
            "table_counts": counts,
            "data_dir": DATA_DIR,
        }

    except Exception as exc:
        logger.exception("Prep pipeline failed: %s", exc)
        raise
    finally:
        if con is not None:
            con.close()
            logger.info("DuckDB connection closed")

        if cleanup:
            cleanup_temp_files(logger)


def main() -> None:
    """Batch entry point for script execution."""
    run_prep(cleanup=True, export_parquet_files=True)


if __name__ == "__main__":
    main()
