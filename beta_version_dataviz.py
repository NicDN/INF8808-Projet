#!/usr/bin/env python3
"""
Interactive Plotly outputs (two HTML files in this folder):

1) heatmap_interactive_beta_plotly.html — 2D density (age vs height/weight) with dropdown.
2) story_dominant_side_proportion_plotly.html — pie chart (tremor stronger on dominant vs non-dominant hand).

Data (bundled next to this script): preprocessed/file_list.csv and movement/timeseries/*.txt
under scripts/beta-release/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots


THEME = {
    "background_color": "#ffffff",
    "font_family": "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif",
    "accent_font_family": "Inter, 'Segoe UI', system-ui, sans-serif",
    "dark_color": "#2A2B2E",
    "label_font_size": 14,
    "title_font_size": 20,
    "label_background_color": "#ffffff",
    # Same as INF8808-TP/tp3/code/template.py (heatmap example)
    "colorscale": "Bluyl",
}


def pie_slice_colors() -> list[str]:
    """Two distinct colors from the same Bluyl scale as the heatmap (low vs high stops)."""
    return sample_colorscale(THEME["colorscale"], [0.42, 0.94])

DISPLAY = {
    "parkinson": "Parkinson",
    "healthy": "Healthy",
    "age_at_diagnosis": "Age At Diagnosis",
    "age": "Age",
    "height": "Height",
    "weight": "Weight",
    "count": "Nombre De Patients",
}


def gyro_magnitude_from_file(filepath: Path) -> float:
    """Return mean gyroscope magnitude (rad/s) for one timeseries file."""
    data = np.loadtxt(filepath, dtype=np.float32, delimiter=",")
    if data.size == 0 or data.ndim != 2:
        return float("nan")
    gyro = data[:, 4:7]
    gyro_mag = np.sqrt(np.sum(gyro**2, axis=1))
    return float(np.nanmean(gyro_mag))


def parse_timeseries_filename(name: str) -> tuple[str | None, str | None]:
    """Parse '001_Relaxed_LeftWrist.txt' -> ('001', 'LeftWrist')."""
    parts = Path(name).stem.split("_")
    if len(parts) < 3:
        return None, None
    return parts[0], "_".join(parts[2:])


def make_dominant_side_pie(df_meta: pd.DataFrame, data_root: Path) -> go.Pie | None:
    """Create pie chart of worse tremor side for Parkinson's patients."""
    df_patients = df_meta[["id", "label", "handedness"]].copy()
    df_patients["id_str"] = df_patients["id"].astype(str).str.zfill(3)
    df_patients = df_patients[df_patients["label"] == 1].set_index("id_str")

    rows = []
    timeseries_dir = data_root / "movement" / "timeseries"
    for fpath in sorted(timeseries_dir.glob("*.txt")):
        subject_id, wrist = parse_timeseries_filename(fpath.name)
        if subject_id is None or subject_id not in df_patients.index:
            continue
        handedness = df_patients.loc[subject_id, "handedness"]
        is_dominant = (handedness == "right" and wrist == "RightWrist") or (
            handedness == "left" and wrist == "LeftWrist"
        )
        rows.append(
            {
                "subject_id": subject_id,
                "hand_type": "Dominant" if is_dominant else "Non-dominant",
                "gyro_magnitude": gyro_magnitude_from_file(fpath),
            }
        )

    tremor = pd.DataFrame(rows)
    if tremor.empty:
        return None

    by_hand = tremor.groupby(["subject_id", "hand_type"], as_index=False)["gyro_magnitude"].mean()
    wide = by_hand.pivot(index="subject_id", columns="hand_type", values="gyro_magnitude")
    if "Dominant" not in wide.columns or "Non-dominant" not in wide.columns:
        return None
    wide = wide.dropna(subset=["Dominant", "Non-dominant"])
    if wide.empty:
        return None

    diff = wide["Dominant"] - wide["Non-dominant"]
    worse = np.where(diff > 0, "Main dominante", np.where(diff < 0, "Main non dominante", "Similaire"))
    worse = pd.Series(worse)
    worse = worse[worse != "Similaire"]
    if worse.empty:
        return None

    counts = worse.value_counts()
    labels = ["Main dominante", "Main non dominante"]
    values = [int(counts.get(label, 0)) for label in labels]
    return go.Pie(
        labels=labels,
        values=values,
        sort=False,
        marker=dict(
            colors=pie_slice_colors(),
            line=dict(color=THEME["dark_color"], width=1.2),
        ),
        textinfo="label+percent",
        textposition="auto",
        textfont=dict(
            family=THEME["font_family"],
            size=THEME["label_font_size"],
            color=THEME["background_color"],
        ),
        insidetextorientation="horizontal",
        hovertemplate=(
            f"<b>%{{label}}</b><br>"
            f"<b>{DISPLAY['count']}</b>: %{{value}}<br>"
            "Pourcentage: %{percent}<extra></extra>"
        ),
        showlegend=True,
    )


def create_custom_theme() -> None:
    """Register custom Plotly template (shared by heatmap and pie)."""
    custom_template = go.layout.Template(
        layout=go.Layout(
            font=dict(
                family=THEME["font_family"],
                color=THEME["dark_color"],
                size=THEME["label_font_size"],
            ),
            paper_bgcolor=THEME["background_color"],
            plot_bgcolor=THEME["background_color"],
            hoverlabel=dict(
                font=dict(
                    family=THEME["accent_font_family"],
                    size=THEME["label_font_size"],
                    color=THEME["dark_color"],
                ),
                bgcolor=THEME["label_background_color"],
            ),
            hovermode="closest",
            xaxis=dict(tickangle=-45),
        )
    )
    custom_template.data.heatmap = [go.Heatmap(colorscale=THEME["colorscale"])]
    pio.templates["custom_theme"] = custom_template
    pio.templates.default = "plotly_white+custom_theme"


def make_trace(df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str) -> go.Histogram2d:
    """Create one Histogram2d trace."""
    data = df.dropna(subset=[x_col, y_col])
    return go.Histogram2d(
        x=data[x_col],
        y=data[y_col],
        nbinsx=30,
        nbinsy=30,
        coloraxis="coloraxis",
        hovertemplate=(
            f"<b>{x_label}</b>: %{{x}}<br>"
            f"<b>{y_label}</b>: %{{y}}<br>"
            f"<b>{DISPLAY['count']}</b>: %{{z}}<extra></extra>"
        ),
        visible=False,
    )


def build_title(group_key: str, y_key: str) -> str:
    """Return centered bold title text."""
    if group_key == "parkinson":
        if y_key == "height":
            text = "Analyse de la taille et de l'âge au diagnostic (patients parkinsoniens)"
        else:
            text = "Analyse du poids et de l'âge au diagnostic (patients parkinsoniens)"
    else:
        if y_key == "height":
            text = "Analyse de la taille et de l'âge (participants sains)"
        else:
            text = "Analyse du poids et de l'âge (participants sains)"
    return f"<b>{text}</b>"


def build_pie_title_html() -> str:
    """Centered title matching heatmap style (bold main line + subtitle)."""
    return (
        "<b>Où les tremblements sont-ils les plus intenses ?</b>"
        "<br><span style='font-size:15px;font-weight:normal;'>Patients parkinsoniens</span>"
    )


def write_heatmap_html(out_dir: Path, df: pd.DataFrame) -> Path:
    """Single interactive heatmap figure with dropdown."""
    parkinson = df[df["label"] == 1]
    healthy = df[df["label"] == 0]

    traces = [
        make_trace(
            parkinson,
            "age_at_diagnosis",
            "height",
            DISPLAY["age_at_diagnosis"],
            DISPLAY["height"],
        ),
        make_trace(
            parkinson,
            "age_at_diagnosis",
            "weight",
            DISPLAY["age_at_diagnosis"],
            DISPLAY["weight"],
        ),
        make_trace(
            healthy,
            "age",
            "height",
            DISPLAY["age"],
            DISPLAY["height"],
        ),
        make_trace(
            healthy,
            "age",
            "weight",
            DISPLAY["age"],
            DISPLAY["weight"],
        ),
    ]
    traces[0].visible = True

    # Two rows: top strip = title + controls only; bottom = heatmap (no overlap).
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.22, 0.78],
        vertical_spacing=0.04,
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
    )

    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="none", showlegend=False, hoverinfo="skip"),
        row=1,
        col=1,
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=1, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], row=1, col=1)

    for t in traces:
        fig.add_trace(t, row=2, col=1)

    fig.update_layout(
        template="plotly_white+custom_theme",
        dragmode=False,
        height=640,
        title=dict(text=""),
        margin=dict(l=72, r=48, t=28, b=72),
        coloraxis=dict(
            colorscale=THEME["colorscale"],
            colorbar=dict(
                title=dict(text=DISPLAY["count"], font=dict(family=THEME["font_family"], size=THEME["label_font_size"])),
                tickfont=dict(family=THEME["font_family"], color=THEME["dark_color"]),
            ),
        ),
        xaxis2=dict(title=DISPLAY["age_at_diagnosis"]),
        yaxis2=dict(title=DISPLAY["height"]),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                x=0.02,
                # Figure coords (0=bottom, 1=top): place below "Choisir la vue" so label + menu don't overlap.
                y=0.852,
                xanchor="left",
                yanchor="top",
                showactive=True,
                font=dict(family=THEME["font_family"], size=THEME["label_font_size"], color=THEME["dark_color"]),
                bgcolor=THEME["background_color"],
                bordercolor=THEME["dark_color"],
                buttons=[
                    dict(
                        label="Parkinson - Height",
                        method="update",
                        args=[
                            {"visible": [True, False, False, False]},
                            {
                                "annotations[0].text": build_title("parkinson", "height"),
                                "xaxis2.title.text": DISPLAY["age_at_diagnosis"],
                                "yaxis2.title.text": DISPLAY["height"],
                            },
                        ],
                    ),
                    dict(
                        label="Parkinson - Weight",
                        method="update",
                        args=[
                            {"visible": [False, True, False, False]},
                            {
                                "annotations[0].text": build_title("parkinson", "weight"),
                                "xaxis2.title.text": DISPLAY["age_at_diagnosis"],
                                "yaxis2.title.text": DISPLAY["weight"],
                            },
                        ],
                    ),
                    dict(
                        label="Healthy - Height",
                        method="update",
                        args=[
                            {"visible": [False, False, True, False]},
                            {
                                "annotations[0].text": build_title("healthy", "height"),
                                "xaxis2.title.text": DISPLAY["age"],
                                "yaxis2.title.text": DISPLAY["height"],
                            },
                        ],
                    ),
                    dict(
                        label="Healthy - Weight",
                        method="update",
                        args=[
                            {"visible": [False, False, False, True]},
                            {
                                "annotations[0].text": build_title("healthy", "weight"),
                                "xaxis2.title.text": DISPLAY["age"],
                                "yaxis2.title.text": DISPLAY["weight"],
                            },
                        ],
                    ),
                ],
            ),
        ],
    )

    fig.add_annotation(
        text=build_title("parkinson", "height"),
        x=0.5,
        y=0.92,
        xanchor="center",
        yanchor="top",
        showarrow=False,
        font=dict(
            family=THEME["font_family"],
            size=THEME["title_font_size"],
            color=THEME["dark_color"],
        ),
        row=1,
        col=1,
    )
    fig.add_annotation(
        text="Choisir la vue",
        x=0.02,
        y=0.44,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        font=dict(family=THEME["font_family"], size=THEME["label_font_size"], color=THEME["dark_color"]),
        row=1,
        col=1,
    )

    out_path = out_dir / "heatmap_interactive_beta_plotly.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def write_pie_html(out_dir: Path, df: pd.DataFrame, data_root: Path) -> Path:
    """Standalone pie chart HTML with same theme as heatmap."""
    pie_trace = make_dominant_side_pie(df, data_root)
    fig = go.Figure()
    if pie_trace is not None:
        fig.add_trace(pie_trace)
    else:
        fig.add_annotation(
            text="Données insuffisantes pour ce graphique (vérifiez movement/timeseries et file_list.csv).",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(family=THEME["font_family"], size=THEME["label_font_size"], color=THEME["dark_color"]),
        )

    fig.update_layout(
        template="plotly_white+custom_theme",
        dragmode=False,
        height=560,
        margin=dict(l=48, r=48, t=100, b=100),
        title=dict(
            text=build_pie_title_html(),
            x=0.5,
            xanchor="center",
            font=dict(
                family=THEME["font_family"],
                size=THEME["title_font_size"],
                color=THEME["dark_color"],
            ),
        ),
        showlegend=pie_trace is not None,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(family=THEME["font_family"], size=THEME["label_font_size"], color=THEME["dark_color"]),
            bgcolor="rgba(255,255,255,0)",
        ),
    )

    out_path = out_dir / "story_dominant_side_proportion_plotly.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    return out_path


def main() -> None:
    create_custom_theme()

    # Data live alongside this script (scripts/beta-release/preprocessed, .../movement/timeseries).
    data_root = Path(__file__).resolve().parent
    out_dir = data_root
    df = pd.read_csv(data_root / "preprocessed" / "file_list.csv")

    p1 = write_heatmap_html(out_dir, df)
    p2 = write_pie_html(out_dir, df, data_root)
    print(f"Saved: {p1}")
    print(f"Saved: {p2}")


if __name__ == "__main__":
    main()
