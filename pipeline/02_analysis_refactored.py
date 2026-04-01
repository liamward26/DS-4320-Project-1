import logging
from typing import Any, Dict, List

import duckdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DB_PATH = "data/movielens.db"


def configure_logging(log_path: str = "analysis.log") -> logging.Logger:
    """Configure and return the analysis logger."""
    logger = logging.getLogger("analysis_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def load_analysis_dataframe(
    db_path: str = DB_PATH,
    logger: logging.Logger | None = None,
    dev_sample_size: int | None = None,
) -> pd.DataFrame:
    """Load the merged analysis dataframe from DuckDB."""
    con = duckdb.connect(db_path)

    query = """
    SELECT
        r.userId,
        r.movieId,
        r.rating,
        r.timestamp,
        m.genres,
        u.age_group,
        u.sex
    FROM ratings r
    JOIN movies m ON r.movieId = m.movieId
    JOIN users_demo u ON r.userId = u.userId
    """

    df = con.execute(query).fetchdf()
    con.close()

    if dev_sample_size is not None and len(df) > dev_sample_size:
        df = df.sample(n=dev_sample_size, random_state=4320)

    if logger:
        logger.info("Loaded analysis dataframe with %s rows", len(df))

    return df


def expand_genres(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pipe-separated genres into binary indicator columns."""
    genre_dummies = df["genres"].str.get_dummies(sep="|").add_prefix("genre_")
    df = df.drop(columns=["genres"])
    df = pd.concat([df, genre_dummies], axis=1)
    return df


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add user/movie aggregate rating features used by both models."""
    movie_stats = df.groupby("movieId")["rating"].agg(
        movie_mean_rating="mean",
        movie_rating_count="count"
    ).reset_index()

    user_stats = df.groupby("userId")["rating"].agg(
        user_mean_rating="mean",
        user_rating_count="count"
    ).reset_index()

    df = df.merge(movie_stats, on="movieId", how="left")
    df = df.merge(user_stats, on="userId", how="left")
    return df


def build_model(
    scaled_numeric_features: List[str],
    passthrough_binary_features: List[str],
    categorical_features: List[str],
) -> Pipeline:
    """
    Build a Ridge regression model with preprocessing.

    Continuous features are standardized, genre indicators are passed through,
    and demographic categories are one-hot encoded.
    """
    transformers = [
        (
            "scaled_num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            scaled_numeric_features,
        ),
        (
            "binary",
            "passthrough",
            passthrough_binary_features,
        ),
    ]

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0)),
    ])


def get_model_coefficients(
    model: Pipeline,
    scaled_numeric_features: List[str],
    passthrough_binary_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """Extract model coefficients and align them to transformed feature names."""
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names: List[str] = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "scaled_num":
            feature_names.extend(cols)
        elif name == "binary":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            encoded = ohe.get_feature_names_out(cols)
            feature_names.extend(encoded)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": regressor.coef_,
    })

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    return coef_df.drop(columns=["abs_coefficient"])


def run_analysis(
    db_path: str = DB_PATH,
    dev_sample_size: int | None = None,
    test_size: float = 0.2,
    random_state: int = 4320,
    output_dir: str = "data",
    log_path: str = "analysis.log",
) -> Dict[str, Any]:
    """
    Run the full modeling workflow and return results/data objects.

    Returns results, coefficients, predictions, and the train/test dataframes so the
    notebook can display them directly.
    """
    logger = configure_logging(log_path)
    logger.info("Starting analysis")

    df = load_analysis_dataframe(db_path=db_path, logger=logger, dev_sample_size=dev_sample_size)
    df = expand_genres(df)
    logger.info("Expanded genres into binary columns")

    df = add_aggregate_features(df)

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    logger.info("Train rows: %s, Test rows: %s", len(train_df), len(test_df))

    y_train = train_df["rating"]
    y_test = test_df["rating"]

    genre_cols = sorted([c for c in df.columns if c.startswith("genre_")])

    scaled_numeric_features = [
        "movie_mean_rating",
        "movie_rating_count",
        "user_mean_rating",
        "user_rating_count",
    ]

    binary_genre_features = genre_cols
    baseline_features = scaled_numeric_features + binary_genre_features
    enhanced_features = baseline_features + ["age_group", "sex"]

    baseline_model = build_model(
        scaled_numeric_features=scaled_numeric_features,
        passthrough_binary_features=binary_genre_features,
        categorical_features=[],
    )

    enhanced_model = build_model(
        scaled_numeric_features=scaled_numeric_features,
        passthrough_binary_features=binary_genre_features,
        categorical_features=["age_group", "sex"],
    )

    logger.info("Training baseline model")
    baseline_model.fit(train_df[baseline_features], y_train)

    logger.info("Training enhanced model")
    enhanced_model.fit(train_df[enhanced_features], y_train)

    baseline_pred = baseline_model.predict(test_df[baseline_features])
    enhanced_pred = enhanced_model.predict(test_df[enhanced_features])

    results = pd.DataFrame({
        "model": ["baseline", "enhanced"],
        "RMSE": [
            np.sqrt(mean_squared_error(y_test, baseline_pred)),
            np.sqrt(mean_squared_error(y_test, enhanced_pred)),
        ],
        "MAE": [
            mean_absolute_error(y_test, baseline_pred),
            mean_absolute_error(y_test, enhanced_pred),
        ],
        "R2": [
            r2_score(y_test, baseline_pred),
            r2_score(y_test, enhanced_pred),
        ],
    })

    results.to_csv(f"{output_dir}/model_comparison.csv", index=False)

    baseline_coef = get_model_coefficients(
        baseline_model,
        scaled_numeric_features=scaled_numeric_features,
        passthrough_binary_features=binary_genre_features,
        categorical_features=[],
    )
    enhanced_coef = get_model_coefficients(
        enhanced_model,
        scaled_numeric_features=scaled_numeric_features,
        passthrough_binary_features=binary_genre_features,
        categorical_features=["age_group", "sex"],
    )

    baseline_coef.to_csv(f"{output_dir}/baseline_coefficients.csv", index=False)
    enhanced_coef.to_csv(f"{output_dir}/enhanced_coefficients.csv", index=False)

    prediction_output = test_df[["userId", "movieId", "rating"]].copy()
    prediction_output["baseline_pred"] = baseline_pred
    prediction_output["enhanced_pred"] = enhanced_pred
    prediction_output.to_csv(f"{output_dir}/test_predictions.csv", index=False)

    logger.info("Analysis completed successfully")
    logger.info("\n%s", results.to_string(index=False))

    return {
        "results": results,
        "baseline_coefficients": baseline_coef,
        "enhanced_coefficients": enhanced_coef,
        "predictions": prediction_output,
        "train_df": train_df,
        "test_df": test_df,
    }


def main() -> None:
    """Batch entry point for script execution."""
    output = run_analysis()
    print("\nModel Comparison:\n")
    print(output["results"])
    print("\nTop Baseline Features:")
    print(output["baseline_coefficients"].head(10))
    print("\nTop Enhanced Features:")
    print(output["enhanced_coefficients"].head(15))


if __name__ == "__main__":
    main()
