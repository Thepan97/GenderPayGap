import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# -----------------------------
# 1. Load & prepare the data
# -----------------------------
# If the Excel file is in the same folder as app.py
DATA_PATH = "GPG_Combined v2.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name="GPG_Combined")

# Ensure Year is numeric (e.g. 2017, 2018, …)
df["YearNum"] = pd.to_datetime(df["Year"]).dt.year

# Convenience: use a shorter name for the main metric
GPG_COL = "GenderPayGap_HourlyPay_Mean_Percent"

# Convert to numeric (coerce non-numeric to NaN)
df[GPG_COL] = pd.to_numeric(df[GPG_COL], errors="coerce")

# Drop rows with no GPG value
df = df.dropna(subset=[GPG_COL])

# List of available years
year_min = int(df["YearNum"].min())
year_max = int(df["YearNum"].max())
year_marks = {int(y): str(int(y)) for y in sorted(df["YearNum"].unique())}

# -----------------------------
# 2. Build the Dash app
# -----------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"font-family": "Segoe UI, sans-serif", "margin": "20px"},
    children=[
        # html.H2("Change in Gender Pay Gap Over Time"),
        html.H2(
            "Change in Gender Pay Gap Over Time",
            style={
                "background-color": "#26375C",
                "color": "white",
                "padding": "10px",
                "text-align": "center",
                "border-radius": "4px",
                "margin-bottom": "20px",
            },
        ),


        html.Div(
            style={"display": "flex", "gap": "40px", "align-items": "center"},
            children=[
                html.Div(
                    children=[
                        html.Label("Select year range (earlier & later):"),
                        dcc.RangeSlider(
                            id="year-range",
                            min=year_min,
                            max=year_max,
                            step=1,
                            value=[year_min, year_max],
                            marks=year_marks,
                            allowCross=False,
                        ),
                    ],
                    style={"flex": "2"},
                ),
                html.Div(
                    children=[
                        html.Label("Profit type:"),
                        dcc.Dropdown(
                            id="profit-filter",
                            options=[
                                {"label": "All", "value": "ALL"},
                                {"label": "For profit", "value": "For profit"},
                                {"label": "Non-profit", "value": "Non-profit"},
                            ],
                            value="ALL",
                            clearable=False,
                            style={"width": "200px"},
                        ),
                    ],
                    style={"flex": "1"},
                ),
            ],
        ),

        html.Div(style={"height": "30px"}),

        dcc.Graph(id="gpg-dumbbell", style={"height": "700px"}),

        html.Div(
            id="legend-text",
            style={"margin-top": "10px", "font-size": "14px", "text-align": "center"},
        ),
    ],
)

# -----------------------------
# 3. Callback: update dumbbell
# -----------------------------
@app.callback(
    Output("gpg-dumbbell", "figure"),
    Output("legend-text", "children"),
    Input("year-range", "value"),
    Input("profit-filter", "value"),
)
def update_dumbbell(year_range, profit_filter):
    start_year, end_year = [int(x) for x in year_range]

    if start_year == end_year:
        return go.Figure(), "Please select two different years."

    # Start from full df
    dff = df.copy()

    # ✅ Filter by profit, if not “ALL”
    if profit_filter != "ALL":
        dff = dff[dff["Profit"] == profit_filter]

    # ✅ Then filter by the two selected years
    dff = dff[dff["YearNum"].isin([start_year, end_year])]

    # Pivot: one row per organisation with both years
    pivot = (
        dff.pivot_table(
            index="Organisation",
            columns="YearNum",
            values=GPG_COL,
            aggfunc="median",
        )
        .reset_index()
    )

    if start_year not in pivot.columns or end_year not in pivot.columns:
        return go.Figure(), "No data available for the selected years."

    pivot = pivot.dropna(subset=[start_year, end_year])

    if pivot.empty:
        return go.Figure(), "No organisations have data in both selected years."

    pivot = pivot.sort_values(end_year)
    pivot["y_pos"] = range(len(pivot))
    y_vals = pivot["y_pos"]
    org_labels = pivot["Organisation"]

    fig = go.Figure()

    # Lines
    for _, row in pivot.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row[start_year], row[end_year]],
                y=[row["y_pos"], row["y_pos"]],
                mode="lines",
                line=dict(color="#CCCCCC", width=2),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Earlier year (turquoise)
    fig.add_trace(
        go.Scatter(
            x=pivot[start_year],
            y=y_vals,
            mode="markers",
            marker=dict(color="#00B9C4", size=10),
            name=f"GPG {start_year}",
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{start_year} GPG: %{{x:.2f}}%<extra></extra>"
            ),
            text=org_labels,
        )
    )

    # Later year (dark blue)
    fig.add_trace(
        go.Scatter(
            x=pivot[end_year],
            y=y_vals,
            mode="markers",
            marker=dict(color="#002B5C", size=10),
            name=f"GPG {end_year}",
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{end_year} GPG: %{{x:.2f}}%<extra></extra>"
            ),
            text=org_labels,
        )
    )

    fig.update_yaxes(
        tickmode="array",
        tickvals=list(y_vals),
        ticktext=list(org_labels),
        autorange="reversed",
        title_text=None,
    )

    fig.update_xaxes(
        zeroline=True,
        zerolinecolor="#999999",
        zerolinewidth=1,
        title_text="Gender pay gap (%)",
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=200, r=40, t=50, b=40),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08),
        font=dict(family="Segoe UI, sans-serif", size=12),
        # title=dict(
        #     text="Change in Gender Pay Gap Over Time",
        #     x=0.5,
        #     xanchor="center",
        # ),
    )
    
    
    legend_text = (
        f"Dark blue represents {end_year} (later year) and "
        f"turquoise represents {start_year} (earlier year)."
    )

    fig.add_annotation(
    text="←IN FAVOUR OF WOMEN",
    xref="paper", yref="paper",
    x=0.0, y=1.08,  # left top
    showarrow=False,
    font=dict(size=12, color="#444", family="Segoe UI, sans-serif"),
    align="left"
    )

    fig.add_annotation(
        text="IN FAVOUR OF MEN→",
        xref="paper", yref="paper",
        x=1.0, y=1.08,  # right top
        showarrow=False,
        font=dict(size=12, color="#444", family="Segoe UI, sans-serif"),
        align="right"
    )

    fig.update_xaxes(
    zeroline=True,
    zerolinecolor="#999999",
    zerolinewidth=1,
    title_text="Gender Pay Gap (%) — lower values favour women, higher favour men"
    )

    return fig, legend_text

if __name__ == "__main__":
    app.run(debug=True)
