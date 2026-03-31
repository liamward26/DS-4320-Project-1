import os
import zipfile
import logging
import requests
import duckdb
import numpy as np
import pandas as pd

DATA_DIR = "data"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
ZIP_PATH = os.path.join(DATA_DIR, "ml-32m.zip")
DB_PATH = os.path.join(DATA_DIR, "movielens.db")
RANDOM_SEED = 4320

os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

rng = np.random.default_rng(RANDOM_SEED)


def download_and_extract() -> None:
    logging.info("Starting dataset download")

    with requests.get(ZIP_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(ZIP_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    logging.info("Download complete")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith(".csv"):
                filename = os.path.basename(member)
                out_path = os.path.join(DATA_DIR, filename)
                with zip_ref.open(member) as source, open(out_path, "wb") as target:
                    target.write(source.read())

    logging.info("CSV extraction complete")


def validate_raw_files() -> None:
    required = ["ratings.csv", "movies.csv", "tags.csv", "links.csv"]
    for filename in required:
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing expected file: {path}")
        logging.info(f"Validated raw file: {path}")


def generate_synthetic_demographics() -> None:
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    user_ids = np.sort(ratings["userId"].dropna().unique())

    age_groups = ["18-24", "25-34", "35-44", "45-54", "55+"]
    age_probs = [0.12, 0.18, 0.17, 0.16, 0.37]

    sex_values = ["Male", "Female"]
    sex_probs = [0.49, 0.51]

    users_demo = pd.DataFrame({
        "userId": user_ids,
        "age_group": rng.choice(age_groups, size=len(user_ids), p=age_probs),
        "sex": rng.choice(sex_values, size=len(user_ids), p=sex_probs),
        "synth_source": "ACS_sampled"
    })

    users_demo.to_csv(os.path.join(DATA_DIR, "users_demo.csv"), index=False)
    logging.info("Synthetic demographics created")


def create_adjusted_ratings() -> None:
    ratings = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
    users_demo = pd.read_csv(os.path.join(DATA_DIR, "users_demo.csv"))

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
    adjusted.to_csv(os.path.join(DATA_DIR, "ratings_adjusted.csv"), index=False)
    logging.info("Adjusted ratings created")


def load_to_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(DB_PATH)

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

    logging.info("Loaded all core tables into DuckDB")
    return con


def validate_database(con: duckdb.DuckDBPyConnection) -> None:
    tables = ["ratings", "movies", "tags", "links", "users_demo"]
    for table in tables:
        count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logging.info(f"{table} row count: {count}")

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

    ratings_count = con.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]

    if joined != ratings_count:
        raise ValueError(
            f"Join validation failed: ratings={ratings_count}, joined={joined}"
        )

    logging.info("Database validation passed")


def export_parquet(con: duckdb.DuckDBPyConnection) -> None:
    for table in ["ratings", "movies", "tags", "links", "users_demo"]:
        out_path = os.path.join(DATA_DIR, f"{table}.parquet")
        con.execute(f"COPY {table} TO '{out_path}' (FORMAT PARQUET)")
        logging.info(f"Exported {table} to parquet")


def cleanup_temp_files() -> None:
    temp_files = [
        os.path.join(DATA_DIR, "ratings.csv"),
        os.path.join(DATA_DIR, "ratings_adjusted.csv"),
        os.path.join(DATA_DIR, "movies.csv"),
        os.path.join(DATA_DIR, "tags.csv"),
        os.path.join(DATA_DIR, "links.csv"),
        os.path.join(DATA_DIR, "users_demo.csv"),
        ZIP_PATH,
    ]

    for path in temp_files:
        try:
            if os.path.exists(path):
                os.remove(path)
                logging.info(f"Deleted temp file: {path}")
        except Exception as e:
            logging.warning(f"Could not delete {path}: {e}")


def main() -> None:
    con = None
    try:
        download_and_extract()
        validate_raw_files()
        generate_synthetic_demographics()
        create_adjusted_ratings()
        con = load_to_duckdb()
        validate_database(con)
        export_parquet(con)
        logging.info("Prep pipeline completed successfully")
    except Exception as e:
        logging.exception(f"Prep pipeline failed: {e}")
        raise
    finally:
        if con is not None:
            con.close()
            logging.info("DuckDB connection closed")
        cleanup_temp_files()


if __name__ == "__main__":
    main()