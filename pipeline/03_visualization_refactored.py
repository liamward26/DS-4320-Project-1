import os
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_model_comparison_figure(results_path: str = "data/model_comparison.csv") -> go.Figure:
    """Build the RMSE comparison figure used in the report and press release."""
    df = pd.read_csv(results_path)
    df["model"] = df["model"].str.capitalize()
    df["RMSE_label"] = df["RMSE"].map(lambda x: f"{x:.4f} stars")

    fig = px.bar(
        df,
        x="model",
        y="RMSE",
        color="model",
        text="RMSE_label",
        title="Prediction Error by Model",
        labels={
            "model": "",
            "RMSE": "RMSE (prediction error, in stars out of 5)",
        },
        color_discrete_sequence=["#4C78A8", "#72B7B2"],
    )

    y_min = df["RMSE"].min() - 0.001
    y_max = df["RMSE"].max() + 0.001

    fig.update_yaxes(range=[y_min, y_max])

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f} stars out of 5<extra></extra>",
    )

    diff = (
        df.loc[df["model"] == "Enhanced", "RMSE"].values[0]
        - df.loc[df["model"] == "Baseline", "RMSE"].values[0]
    )

    fig.add_annotation(
        x=0.5,
        y=y_max - 0.00015,
        text=f"Difference between models: {diff:.6f} stars",
        showarrow=False,
        font=dict(size=12),
    )

    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        title_x=0.5,
        font=dict(size=13),
        margin=dict(l=60, r=40, t=80, b=60),
    )

    return fig


def save_figure(
    fig: go.Figure,
    png_path: str = "data/model_comparison.png",
    html_path: str = "data/model_comparison.html",
) -> None:
    """Save the figure to disk in HTML and, if possible, PNG format."""
    fig.write_html(html_path)

    try:
        fig.write_image(png_path, scale=3)
    except Exception:
        # HTML export is still useful if kaleido is not installed.
        pass


def run_visualization(
    results_path: str = "data/model_comparison.csv",
    png_path: str = "data/model_comparison.png",
    html_path: str = "data/model_comparison.html",
    show: bool = True,
) -> go.Figure:
    """Create, save, and optionally display the model comparison figure."""
    fig = build_model_comparison_figure(results_path=results_path)
    save_figure(fig, png_path=png_path, html_path=html_path)

    if show:
        fig.show()

    return fig


def main() -> None:
    """Batch entry point for script execution."""
    run_visualization(show=True)


if __name__ == "__main__":
    main()
