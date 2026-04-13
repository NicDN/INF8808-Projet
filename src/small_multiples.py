import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ── Données ────────────────────────────────────────────────────────────────────
df = pd.read_csv('pads_full_features_dataset.csv')

df_plot = df.groupby(["subject", "task"]).agg({
    "age":              "first",
    "condition":        "first",
    "acc_tremor_power": "mean",
    "acc_tremor_ratio": "max",
    "gender":           "first"
}).reset_index()

df_pk = df_plot[df_plot['condition'].isin(["Parkinson's"])]
tasks = df_pk['task'].unique().tolist()

# ── Traduction des tâches ──────────────────────────────────────────────────────
TASK_FR = {
    "RelaxedTask": "Repos (avec calcul mental)",
    "Relaxed":     "Repos total",
    "LiftHold":    "Maintenir le bras levé",
    "HoldWeight":  "Maintenir un poids",
    "StretchHold": "Bras tendus devant soi",
    "TouchIndex":  "Toucher l'index",
    "PointFinger": "Pointer du doigt",
    "TouchNose":   "Toucher le nez",
    "CrossArms":   "Croiser les bras",
    "DrinkGlas":   "Boire dans un verre",
    "Entrainment": "Mouvement rythmé"
}
tasks_fr = [TASK_FR.get(t, t) for t in tasks]

# ── Apparence ──────────────────────────────────────────────────────────────────
COLORS = {
    "male":   "#21A3AC",
    "female": "#D64DBB"
}
MARKER_STYLE = dict(size=7, opacity=0.75, line=dict(width=0.6, color="white"))

# ── Construction du graphique ──────────────────────────────────────────────────
ncols = 3
nrows = -(-len(tasks) // ncols)

fig = make_subplots(
    rows=nrows, cols=ncols,
    subplot_titles=tasks_fr,
    shared_yaxes=False,
    horizontal_spacing=0.06,
    vertical_spacing=0.08
)

shown_legend = set()

for i, task in enumerate(tasks):
    row = i // ncols + 1
    col = i %  ncols + 1
    df_t = df_pk[df_pk['task'] == task]

    for gender, grp in df_t.groupby("gender"):
        show = gender not in shown_legend
        if show:
            shown_legend.add(gender)

        fig.add_trace(go.Scatter(
            x=grp["age"],
            y=grp["acc_tremor_ratio"],
            mode="markers",
            name=gender.capitalize(),
            legendgroup=gender,
            showlegend=show,
            marker=dict(**MARKER_STYLE, color=COLORS[gender]),
            customdata=list(zip(grp["subject"], grp["age"])),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Âge   : %{customdata[1]} ans<br>"
                "Trembl.: <b>%{y:.3f}</b>"
                "<extra></extra>"
            )
        ), row=row, col=col)
for i, task in enumerate(tasks):
    row = i // ncols + 1
    col = i %  ncols + 1
    df_t = df_pk[df_pk['task'] == task]

    for gender, grp in df_t.groupby("gender"):
        show = gender not in shown_legend
        if show:
            shown_legend.add(gender)

        if len(grp) >= 4:
            x_s  = grp["age"].sort_values()
            coef = np.polyfit(grp["age"], grp["acc_tremor_ratio"], 1)
            y_s  = np.polyval(coef, x_s)
            fig.add_trace(go.Scatter(
                x=x_s, y=y_s,
                mode="lines",
                line=dict(color=COLORS[gender], width=2.5, dash="solid"),
                legendgroup=gender,
                showlegend=False,
                hoverinfo="skip"
            ), row=row, col=col)

# ── Mise en page globale ───────────────────────────────────────────────────────
fig.update_layout(
    title=dict(
        text="Taux de tremblement par âge et par tâche — Parkinson",
        font=dict(size=18, family="Arial, sans-serif"),
        x=0.01, xanchor="left"
    ),
    height=950,
    width=1150,
    template="plotly_white",
    legend=dict(
        title=dict(text="Genre", font=dict(size=12)),
        orientation="v",
        x=1.01, y=1,
        borderwidth=1,
        bordercolor="#E0E0E0"
    ),
    font=dict(family="Arial, sans-serif", size=11),
    margin=dict(t=80, r=120, b=60, l=80),
    plot_bgcolor="white",
    paper_bgcolor="white"
)

# ── Axes : style uniforme sur tous les sous-graphiques ─────────────────────────
AXIS_STYLE = dict(
    showgrid=True, gridcolor="#F0F0F0", gridwidth=1,
    showline=True, linecolor="#CCCCCC",
    zeroline=False
)

fig.update_xaxes(**AXIS_STYLE)
fig.update_yaxes(**AXIS_STYLE)

for i in range(1, nrows * ncols + 1):
    show_xlab = (i > (nrows - 1) * ncols)
    show_ylab = (i % ncols == 1)

    fig.update_layout(**{
        f"xaxis{'' if i == 1 else i}": dict(title_text="Âge (ans)" if show_xlab else ""),
        f"yaxis{'' if i == 1 else i}": dict(title_text="Taux de tremblement" if show_ylab else ""),
    })

# ── Style des titres de sous-graphiques ───────────────────────────────────────
fig.for_each_annotation(lambda a: a.update(font=dict(size=12, color="#444")))

# ── Export ─────────────────────────────────────────────────────────────────────
fig.write_html("taux_tremblement_small_multiples.html", include_plotlyjs="cdn")
fig.show()