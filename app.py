import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go

# -----------------------------
# 1. Load & prepare the data
# -----------------------------
DATA_PATH = "GPG_Combined v2.xlsx"

df = pd.read_excel(DATA_PATH, sheet_name="GPG_Combined")

# Ensure Year is numeric (e.g. 2017, 2018, …)
df["YearNum"] = pd.to_datetime(df["Year"]).dt.year

GPG_COL = "GenderPayGap_HourlyPay_Mean_Percent"
df[GPG_COL] = pd.to_numeric(df[GPG_COL], errors="coerce")
df = df.dropna(subset=[GPG_COL])

# Earliest and latest years
year_min = int(df["YearNum"].min())
year_max = int(df["YearNum"].max())  # fixed later year

# Marks for the earlier-year slider (exclude latest year)
earlier_year_marks = {
    int(y): str(int(y))
    for y in sorted(df["YearNum"].unique())
    if int(y) < year_max
}

# Compute fixed x-axis limits across all years
x_min = df[GPG_COL].min()
x_max = df[GPG_COL].max()

# Add a small margin (so points aren't too close to edges)
x_padding = (x_max - x_min) * 0.05
x_range = [x_min - x_padding, x_max + x_padding]

# -----------------------------
# 2. Build Dash app
# -----------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"font-family": "Segoe UI, sans-serif", "margin": "20px"},
    children=[
        html.H2(
            "Change in Gender Pay Gap across Organisations",
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
            style={
                "display": "flex",
                "flex-wrap": "wrap",
                "gap": "30px",
                "align-items": "center",
                "margin-bottom": "20px",
            },
            children=[
                html.Div(
                    style={"min-width": "300px", "flex": "1 1 300px"},
                    children=[
                        html.Label(
                            f"Select earlier year (later year fixed at {year_max}):",
                            style={"margin-bottom": "8px", "display": "block"},
                        ),
                        dcc.Slider(
                            id="earlier-year",
                            min=min(earlier_year_marks.keys()),
                            max=max(earlier_year_marks.keys()),
                            step=1,
                            value=min(earlier_year_marks.keys()),
                            marks=earlier_year_marks,
                            included=False,
                        ),
                    ],
                ),
                html.Div(
                    style={"min-width": "220px"},
                    children=[
                        html.Label(
                            "Filter organisations by direction of change:",
                            style={"margin-bottom": "8px", "display": "block"},
                        ),
                        dcc.Dropdown(
                            id="direction-filter",
                            options=[
                                {"label": "All organisations", "value": "all"},
                                {
                                    "label": "Moving in favour of men",
                                    "value": "men",
                                },
                                {
                                    "label": "Moving in favour of women",
                                    "value": "women",
                                },
                            ],
                            value="all",
                            clearable=False,
                            style={"width": "100%"},
                        ),
                    ],
                ),
            ],
        ),
        # Card showing number of organisations
        html.Div(
            id="org-count-card",
            style={
                "backgroundColor": "#F5F7FB",
                "borderRadius": "6px",
                "padding": "10px 16px",
                "display": "inline-block",
                "marginBottom": "10px",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.1)",
                "fontSize": "14px",
            },
            children="Number of organisations shown: -",
        ),
        dcc.Graph(id="gpg-dumbbell"),
        html.Div(
            id="legend-text",
            style={"margin-top": "10px", "font-size": "14px", "text-align": "center"},
        ),
    ],
)

# -----------------------------
# 3. Callback: Update chart
# -----------------------------
@app.callback(
    Output("gpg-dumbbell", "figure"),
    Output("legend-text", "children"),
    Output("org-count-card", "children"),
    Input("earlier-year", "value"),
    Input("direction-filter", "value"),
)
def update_dumbbell(earlier_year, direction_filter):
    start_year = int(earlier_year)
    end_year = year_max  # fixed later year

    if start_year >= end_year:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return (
            fig,
            f"Please select a year earlier than {end_year}.",
            "Number of organisations shown: 0",
        )

    # Filter to the two years of interest
    dff = df[df["YearNum"].isin([start_year, end_year])]

    # Pivot by organisation
    pivot = (
        dff.pivot_table(
            index="Organisation",
            columns="YearNum",
            values=GPG_COL,
            aggfunc="median",
        )
        .reset_index()
    )

    # Ensure later year exists
    if end_year not in pivot.columns:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return (
            fig,
            f"No data available for {end_year}.",
            "Number of organisations shown: 0",
        )

    # Keep all orgs that have data in the later year
    pivot = pivot.dropna(subset=[end_year])
    if pivot.empty:
        fig = go.Figure()
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        return (
            fig,
            f"No organisations have data for {end_year}.",
            "Number of organisations shown: 0",
        )

    # Organisations with both years (for direction + dumbbells)
    if start_year in pivot.columns:
        pivot_with_both = pivot[pivot[start_year].notna()].copy()
    else:
        pivot_with_both = pivot.iloc[0:0].copy()

    # Classify direction of movement where both years exist
    if not pivot_with_both.empty:
        pivot_with_both["change"] = (
            pivot_with_both[end_year] - pivot_with_both[start_year]
        )
        pivot_with_both["move_dir"] = np.where(
            pivot_with_both["change"] > 0,
            "men",
            np.where(pivot_with_both["change"] < 0, "women", "none"),
        )
    else:
        pivot_with_both["move_dir"] = []

    # Merge movement direction back to full pivot (orgs without start_year stay NaN)
    if "move_dir" in pivot_with_both.columns:
        pivot = pivot.merge(
            pivot_with_both[["Organisation", "move_dir"]],
            on="Organisation",
            how="left",
        )
    else:
        pivot["move_dir"] = np.nan

    # Apply direction filter
    if direction_filter == "men":
        plot_pivot = pivot[pivot["move_dir"] == "men"].copy()
    elif direction_filter == "women":
        plot_pivot = pivot[pivot["move_dir"] == "women"].copy()
    else:
        plot_pivot = pivot.copy()

    if plot_pivot.empty:
        fig = go.Figure()
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="No organisations match the selected criteria.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14),
                )
            ],
        )
        msg = (
            f"No organisations have data for {start_year} and {end_year} that "
            f"are '{'moving in favour of men' if direction_filter=='men' else 'moving in favour of women'}'."
            if direction_filter in ["men", "women"]
            else "No organisations available for the selected filters."
        )
        return fig, msg, "Number of organisations shown: 0"

    # Sort by latest year
    plot_pivot = plot_pivot.sort_values(end_year)

    # y positions for all orgs in the filtered set
    plot_pivot["y_pos"] = list(range(len(plot_pivot)))
    y_vals = plot_pivot["y_pos"]
    org_labels = plot_pivot["Organisation"]

    fig = go.Figure()

    # Dumbbell lines: only for orgs in this view that have both years
    if start_year in plot_pivot.columns:
        lines_df = plot_pivot[plot_pivot[start_year].notna()].copy()
        for _, row in lines_df.iterrows():
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

    # Earlier year markers (turquoise): only where start_year exists
    if start_year in plot_pivot.columns:
        earlier_points = plot_pivot[plot_pivot[start_year].notna()]
        if not earlier_points.empty:
            fig.add_trace(
                go.Scatter(
                    x=earlier_points[start_year],
                    y=earlier_points["y_pos"],
                    mode="markers",
                    marker=dict(color="#00B9C4", size=10),
                    name=f"GPG {start_year}",
                    hovertemplate="<b>%{text}</b><br>"
                    f"{start_year} GPG: %{{x:.2f}}%<extra></extra>",
                    text=earlier_points["Organisation"],
                )
            )

    # Later year markers (dark blue): for all orgs in this view
    fig.add_trace(
        go.Scatter(
            x=plot_pivot[end_year],
            y=y_vals,
            mode="markers",
            marker=dict(color="#002B5C", size=10),
            name=f"GPG {end_year}",
            hovertemplate="<b>%{text}</b><br>"
            f"{end_year} GPG: %{{x:.2f}}%<extra></extra>",
            text=org_labels,
        )
    )

    # Y axis – labels for each organisation + horizontal gridlines
    fig.update_yaxes(
        tickmode="array",
        tickvals=list(y_vals),
        ticktext=list(org_labels),
        autorange="reversed",
        title_text=None,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.4)",
        gridwidth=1,
        zeroline=False,
    )

    # X axis – fixed range across all updates + vertical gridlines
    fig.update_xaxes(
        range=x_range,
        zeroline=True,
        zerolinecolor="#999999",
        zerolinewidth=1,
        title_text="Gender Pay Gap (%) — lower values favour women, higher favour men",
        showgrid=True,
        gridcolor="rgba(220,220,220,0.3)",
    )

    # Dynamic chart height: more organisations = taller figure
    n_orgs = len(plot_pivot)
    base_height = 250   # space for title, slider, legend, etc.
    row_height = 28     # pixels per organisation row
    fig_height = base_height + row_height * n_orgs

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=250, r=40, t=90, b=40),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08),
        font=dict(family="Segoe UI, sans-serif", size=12),
        height=fig_height,
    )

    # Direction labels (in top margin)
    fig.add_annotation(
        text="←IN FAVOUR OF WOMEN",
        xref="paper",
        yref="paper",
        x=0.0,
        y=1.0,
        yshift=30,
        showarrow=False,
        font=dict(size=12, color="#444", family="Segoe UI, sans-serif"),
        align="left",
    )
    fig.add_annotation(
        text="IN FAVOUR OF MEN→",
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.0,
        yshift=30,
        showarrow=False,
        font=dict(size=12, color="#444", family="Segoe UI, sans-serif"),
        align="right",
    )

    # Legend text
    direction_msg = {
        "all": "all organisations with data in the later year",
        "men": "organisations where the gender pay gap has moved further in favour of men",
        "women": "organisations where the gender pay gap has moved further in favour of women",
    }[direction_filter]

    legend_text = (
        f"Dark blue represents {end_year} (later year, fixed). "
        f"Turquoise represents {start_year} where data is available. "
        f"The current view shows {direction_msg}. "
        f"Organisations without {start_year} data only show the dark-blue marker and "
        f"are not included in the 'moving in favour' filters."
    )

    org_count_text = f"Number of organisations shown: {n_orgs}"

    return fig, legend_text, org_count_text


if __name__ == "__main__":
    app.run(debug=True)
