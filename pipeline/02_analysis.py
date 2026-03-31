import duckdb
import pandas as pd
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

DB_PATH = "data/movielens.db"


def expand_genres(df: pd.DataFrame) -> pd.DataFrame:
    genre_dummies = df["genres"].str.get_dummies(sep="|").add_prefix("genre_")
    df = df.drop(columns=["genres"])
    df = pd.concat([df, genre_dummies], axis=1)
    return df


def build_model(numeric_features, categorical_features):
    transformers = [
        ("num", SimpleImputer(strategy="median"), numeric_features)
    ]

    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ])


def get_model_coefficients(model, numeric_features, categorical_features):
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["regressor"]

    feature_names = []

    for name, transformer, cols in preprocessor.transformers_:
        if name == "num":
            feature_names.extend(cols)
        elif name == "cat":
            ohe = transformer.named_steps["onehot"]
            encoded = ohe.get_feature_names_out(cols)
            feature_names.extend(encoded)

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": regressor.coef_
    })

    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)
    return coef_df.drop(columns=["abs_coefficient"])


def main():
    logging.info("Starting analysis")

    con = duckdb.connect(DB_PATH)

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

    logging.info(f"Loaded merged analysis dataframe with {len(df)} rows")

    # Optional sample if runtime becomes too large
    if len(df) > 2_000_000:
        df = df.sample(n=2_000_000, random_state=4320)
        logging.info("Sampled merged dataframe to 2,000,000 rows")

    df = expand_genres(df)
    logging.info("Expanded genres into binary columns")

    # Create user/movie aggregates for modeling
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

    # Train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=4320)
    logging.info(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    y_train = train_df["rating"]
    y_test = test_df["rating"]

    genre_cols = sorted([c for c in df.columns if c.startswith("genre_")])

    numeric_base = [
        "movie_mean_rating",
        "movie_rating_count",
        "user_mean_rating",
        "user_rating_count",
    ] + genre_cols

    baseline_features = numeric_base.copy()
    enhanced_features = numeric_base + ["age_group", "sex"]

    baseline_model = build_model(numeric_base, [])
    enhanced_model = build_model(numeric_base, ["age_group", "sex"])

    logging.info("Training baseline model")
    baseline_model.fit(train_df[baseline_features], y_train)

    logging.info("Training enhanced model")
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
        ]
    })

    print(results)
    results.to_csv("data/model_comparison.csv", index=False)

    baseline_coef = get_model_coefficients(baseline_model, numeric_base, [])
    enhanced_coef = get_model_coefficients(enhanced_model, numeric_base, ["age_group", "sex"])

    baseline_coef.to_csv("data/baseline_coefficients.csv", index=False)
    enhanced_coef.to_csv("data/enhanced_coefficients.csv", index=False)

    prediction_output = test_df[["userId", "movieId", "rating"]].copy()
    prediction_output["baseline_pred"] = baseline_pred
    prediction_output["enhanced_pred"] = enhanced_pred
    prediction_output.to_csv("data/test_predictions.csv", index=False)

    logging.info("Analysis completed successfully")


if __name__ == "__main__":
    main()