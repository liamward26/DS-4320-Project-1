import pandas as pd
import plotly.express as px

# Load results
df = pd.read_csv("data/model_comparison.csv")

# Clean labels
df["model"] = df["model"].str.capitalize()

# Add friendly metric label
df["RMSE_label"] = df["RMSE"].map(lambda x: f"{x:.4f} stars")

# Build chart
fig = px.bar(
    df,
    x="model",
    y="RMSE",
    color="model",
    text="RMSE_label",
    title="Prediction Error by Model",
    labels={
        "model": "",
        "RMSE": "RMSE (prediction error, in stars out of 5)"
    },
    color_discrete_sequence=["#4C78A8", "#72B7B2"]
)

# Tight range to make the comparison visible
y_min = df["RMSE"].min() - 0.001
y_max = df["RMSE"].max() + 0.001

fig.update_yaxes(range=[y_min, y_max])

fig.update_traces(
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f} stars out of 5<extra></extra>"
)

diff = df.loc[df["model"] == "Enhanced", "RMSE"].values[0] - df.loc[df["model"] == "Baseline", "RMSE"].values[0]

fig.add_annotation(
    x=0.5,
    y=y_max - 0.00015,
    text=f"Difference between models: {diff:.6f} stars",
    showarrow=False,
    font=dict(size=12)
)

fig.update_layout(
    showlegend=False,
    template="plotly_white",
    title_x=0.5,
    font=dict(size=13),
    margin=dict(l=60, r=40, t=80, b=60)
)

fig.write_image("data/model_comparison.png", scale=3)
fig.write_html("data/model_comparison.html")
fig.show()