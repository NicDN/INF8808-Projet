import pandas as pd
import numpy as np
import json
import os
from scipy.signal import welch, butter, filtfilt

# ============================================================
#  CHEMINS — à adapter selon ton environnement
# ============================================================
PATH_PATIENTS   = r"C:\Users\Marie\Documents\Cours\Poly\INF8808\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\patients"
PATH_MOVEMENT   = r"C:\Users\Marie\Documents\Cours\Poly\INF8808\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\movement"
PATH_TIMESERIES = r"C:\Users\Marie\Documents\Cours\Poly\INF8808\pads-parkinsons-disease-smartwatch-dataset-1.0.0\pads-parkinsons-disease-smartwatch-dataset-1.0.0\movement\timeseries"

OUTPUT_CSV = "rawgraphs_beeswarm.csv"

TASKS_VALIDES = {
    "Relaxed", "RelaxedTask", "LiftHold", "StretchHold", "HoldWeight",
    "PointFinger", "TouchIndex", "TouchNose", "CrossArms", "DrinkGlas", "Entrainment"
}

# ============================================================
#  CALCUL DU TREMOR RATIO (version améliorée)
# ============================================================
def analyze_signal(filepath):
    """
    Calcule le tremor ratio et la fréquence dominante.

    Améliorations vs script précédent :
    - Bande de tremblement élargie à 3-10 Hz (couvre tout le spectre Parkinson)
    - Retrait de la gravité par filtre passe-haut 0.5 Hz (plus robuste que detrend)
    - Bande de référence 0.5-20 Hz pour total_power (exclut DC et bruit haute freq)
    - nperseg adaptatif pour meilleure résolution fréquentielle sur longs signaux
    """
    try:
        data = pd.read_csv(
            filepath,
            usecols=[0, 1, 2, 3],
            names=["Time", "X", "Y", "Z"],
            header=None,
            engine="c"
        )
        data = data.dropna()

        if len(data) < 256:
            return None, None

        # Estimation de la fréquence d'échantillonnage depuis les timestamps
        dt = np.median(np.diff(data["Time"].values))
        fs = round(1.0 / dt) if dt > 0 else 100

        # Magnitude de l'accélération (norme L2)
        acc_mag = np.sqrt(data["X"]**2 + data["Y"]**2 + data["Z"]**2).values

        # Retrait de la gravité : filtre passe-haut Butterworth 0.5 Hz
        # Plus efficace que detrend() qui ne retire que la tendance linéaire
        b, a = butter(4, 0.5 / (fs / 2), btype="high")
        acc_mag = filtfilt(b, a, acc_mag)

        # PSD via Welch — nperseg adaptatif (4 secondes de signal, min 256)
        nperseg = min(len(acc_mag), int(fs * 4))
        nperseg = max(nperseg, 256)
        freqs, psd = welch(acc_mag, fs=fs, nperseg=nperseg)

        # Puissance totale dans la bande 0.5-20 Hz (exclut DC et bruit haute freq)
        ref_mask    = (freqs >= 0.5) & (freqs <= 20.0)
        total_power = np.sum(psd[ref_mask])
        if total_power == 0:
            return None, None

        # Tremor ratio : bande 3-10 Hz (couvre tremblements Parkinson 4-9 Hz)
        tremor_mask  = (freqs >= 3.0) & (freqs <= 10.0)
        tremor_power = np.sum(psd[tremor_mask])
        ratio = tremor_power / total_power

        # Fréquence dominante dans la bande 2-12 Hz
        peak_mask = (freqs >= 2.0) & (freqs <= 12.0)
        dom_freq  = freqs[peak_mask][np.argmax(psd[peak_mask])] if any(peak_mask) else 0.0

        return float(ratio), float(dom_freq)

    except Exception:
        return None, None

# ============================================================
#  PARCOURS DES PATIENTS ET SESSIONS
# ============================================================
rows = []
patient_files = sorted([
    f for f in os.listdir(PATH_PATIENTS)
    if f.startswith("patient_") and f.endswith(".json")
])

print(f"Analyse de {len(patient_files)} patients en cours...")

for filename in patient_files:
    p_id = filename.split("_")[1].replace(".json", "")

    with open(os.path.join(PATH_PATIENTS, filename), "r") as f:
        p_info = json.load(f)

    condition = p_info.get("condition")
    if condition not in ["Parkinson's", "Healthy"]:
        continue

    groupe = "Parkinson" if condition == "Parkinson's" else "Contrôle sain"
    sexe   = "Homme" if str(p_info.get("gender", "")).lower() in ("male", "m") else "Femme"
    age    = p_info.get("age")

    obs_path = os.path.join(PATH_MOVEMENT, f"observation_{p_id}.json")
    if not os.path.exists(obs_path):
        continue

    with open(obs_path, "r") as f:
        obs_data = json.load(f)

    for session in obs_data.get("session", []):
        task = session["record_name"]
        if task not in TASKS_VALIDES:
            continue

        # Résultats par poignet pour cette session
        results = {
            "RightWrist": {"ratio": None, "freq": None},
            "LeftWrist":  {"ratio": None, "freq": None}
        }

        for record in session["records"]:
            wrist     = record["device_location"]  # "LeftWrist" ou "RightWrist"
            file_path = os.path.join(PATH_TIMESERIES, os.path.basename(record["file_name"]))

            if os.path.exists(file_path):
                ratio, freq = analyze_signal(file_path)
                if ratio is not None:
                    results[wrist] = {"ratio": ratio, "freq": freq}

        # On ne garde la ligne que si au moins un poignet est valide
        valid_ratios = [v["ratio"] for v in results.values() if v["ratio"] is not None]
        if not valid_ratios:
            continue

        r_right = results["RightWrist"]["ratio"] or 0.0
        r_left  = results["LeftWrist"]["ratio"]  or 0.0
        best    = "RightWrist" if r_right >= r_left else "LeftWrist"

        rows.append({
            "id":               p_id,
            "groupe":           groupe,
            "task":             task,
            "task_groupe":      f"{groupe} — {task}",
            "max_tremor_ratio": round(max(valid_ratios), 4),
            "ratio_droit":      round(r_right, 4),
            "ratio_gauche":     round(r_left,  4),
            "freq_dominante":   round(results[best]["freq"] or 0.0, 2),
            "age":              age,
            "sexe":             sexe,
        })

if not rows:
    print("Aucune donnée extraite. Vérifie les chemins d'accès.")
    exit()

df = pd.DataFrame(rows)
print(f"   {len(df)} enregistrements | {df['id'].nunique()} patients")

# ============================================================
#  CALCUL DU COHEN'S D PAR TÂCHE
# ============================================================
cohens_d = {}
for task in df["task"].unique():
    sub  = df[df["task"] == task]
    pd_v = sub[sub["groupe"] == "Parkinson"]["max_tremor_ratio"].dropna()
    hc_v = sub[sub["groupe"] == "Contrôle sain"]["max_tremor_ratio"].dropna()
    if len(pd_v) > 5 and len(hc_v) > 5:
        pooled = np.sqrt((pd_v.std()**2 + hc_v.std()**2) / 2)
        cohens_d[task] = round(abs(pd_v.mean() - hc_v.mean()) / (pooled + 1e-10), 3)

# ============================================================
#  ORDRE DES TÂCHES (Cohen's d décroissant)
# ============================================================
ordre_taches = sorted(cohens_d.keys(), key=lambda t: -cohens_d[t])

ordre_categories = []
for t in ordre_taches:
    ordre_categories.append(f"Parkinson — {t}")
    ordre_categories.append(f"Contrôle sain — {t}")

df["cohens_d"]    = df["task"].map(cohens_d).round(3)
df["task_groupe"] = pd.Categorical(df["task_groupe"], categories=ordre_categories, ordered=True)
df = df.sort_values("task_groupe")

# ============================================================
#  EXPORT FINAL
# ============================================================
cols = [
    "id", "groupe", "task", "task_groupe",
    "max_tremor_ratio", "ratio_droit", "ratio_gauche",
    "freq_dominante", "age", "sexe", "cohens_d"
]
df[cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

print(f"\nFichier '{OUTPUT_CSV}' genere avec succes")
print(f"   {len(df)} lignes | {df['id'].nunique()} patients | {df['task'].nunique()} taches")
print(f"\nCohen's d par tache (ordre decroissant) :")
for task in ordre_taches:
    pd_mean = df[(df["task"] == task) & (df["groupe"] == "Parkinson")]["max_tremor_ratio"].mean()
    hc_mean = df[(df["task"] == task) & (df["groupe"] == "Contrôle sain")]["max_tremor_ratio"].mean()
    print(f"  {task:<15}  d = {cohens_d[task]:.3f}   (PD={pd_mean:.3f}  HC={hc_mean:.3f})")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import collections
import os

# ============================================================
#  CONFIG & TRADUCTION
# ============================================================
INPUT_CSV   = "rawgraphs_beeswarm_final5.csv"
OUTPUT_HTML = "visualizations/beeswarm_parkinson_final.html"

COULEUR_PARKINSON = "#D55E00"  # Orange
COULEUR_SAIN      = "#0072B2"  # Bleu

THEME = {
    "background_color": "#ffffff",
    "font_family": "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif",
    "accent_font_family": "Inter, 'Segoe UI', system-ui, sans-serif",
    "dark_color": "#2A2B2E",
    "label_font_size": 14,
    "title_font_size": 20,
    "label_background_color": "#ffffff",
    "colorscale": "Bluyl",
}

BG_TOOLTIP = "#262626"

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

# ============================================================
#  FONCTION SWARM (LOGIQUE DE BINNING)
# ============================================================
def get_swarm_offsets(X_series, fig_width=900, point_size=4, bin_fraction=0.95):
    if len(X_series) == 0:
        return []
    X_series = X_series.copy().sort_values()
    min_x, max_x = 0, 1
    bin_counter = collections.Counter()
    offsets = []
    for x_val in X_series:
        bin_idx = (((fig_width * bin_fraction * (x_val - min_x)) / (max_x - min_x)) // point_size)
        slot = bin_counter[bin_idx]
        bin_counter.update([bin_idx])
        y_val = (slot // 2) * (1 if slot % 2 == 0 else -1)
        offsets.append(y_val)
    return np.array(offsets) * 0.08

# ============================================================
#  PRÉPARATION DES DONNÉES & AXE Y
# ============================================================
df = pd.read_csv(INPUT_CSV)
task_keys = [t for t in TASK_FR.keys() if t in df["task"].unique()]
fig = go.Figure()
y_tick_labels = []
y_tick_vals = []
current_y = 0
POINT_SIZE = 4
FIG_WIDTH = 900

for task_key in reversed(task_keys):
    task_name = TASK_FR[task_key]
    y_sain = current_y
    y_park = current_y + 1

    y_tick_labels.append(task_name)
    y_tick_vals.append(current_y + 0.5)

    # --- TRACE SAIN (Ligne du bas) ---
    subset_sain = df[(df["task"] == task_key) & (df["groupe"] == "Contrôle sain")].copy()
    if not subset_sain.empty:
        subset_sain = subset_sain.sort_values("max_tremor_ratio")
        offsets = get_swarm_offsets(subset_sain["max_tremor_ratio"], fig_width=FIG_WIDTH, point_size=POINT_SIZE)

        hover_sain = [
            f"<b>Groupe :</b> Sain<br>"
            f"<b>Profil :</b> {row['sexe']}, {int(row['age'])} ans<br>"
            f"<b>Part du tremblement :</b> {row['max_tremor_ratio'] * 100:.1f}%<br>"
            f"<b>Poignet Droit :</b> {row['ratio_droit'] * 100:.1f}%<br>"
            f"<b>Poignet Gauche :</b> {row['ratio_gauche'] * 100:.1f}%"
            for _, row in subset_sain.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=subset_sain["max_tremor_ratio"],
            y=[y_sain + off for off in offsets],
            mode="markers",
            name="Sujets sains",
            marker=dict(color=COULEUR_SAIN, size=POINT_SIZE, opacity=0.8, line=dict(width=0.4, color="white")),
            text=hover_sain,
            hoverinfo="text",
            showlegend=(current_y == 0)
        ))

    # --- TRACE PARKINSON (Ligne du haut) ---
    subset_park = df[(df["task"] == task_key) & (df["groupe"] == "Parkinson")].copy()
    if not subset_park.empty:
        subset_park = subset_park.sort_values("max_tremor_ratio")
        offsets = get_swarm_offsets(subset_park["max_tremor_ratio"], fig_width=FIG_WIDTH, point_size=POINT_SIZE)

        hover_park = [
            f"<b>Groupe :</b> Parkinson<br>"
            f"<b>Profil :</b> {row['sexe']}, {int(row['age'])} ans<br>"
            f"<b>Part du tremblement :</b> {row['max_tremor_ratio'] * 100:.1f}%<br>"
            f"<b>Poignet Droit :</b> {row['ratio_droit'] * 100:.1f}%<br>"
            f"<b>Poignet Gauche :</b> {row['ratio_gauche'] * 100:.1f}%"
            for _, row in subset_park.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=subset_park["max_tremor_ratio"],
            y=[y_park + off for off in offsets],
            mode="markers",
            name="Patients Parkinson",
            marker=dict(color=COULEUR_PARKINSON, size=POINT_SIZE, opacity=0.8, line=dict(width=0.4, color="white")),
            text=hover_park,
            hoverinfo="text",
            showlegend=(current_y == 0)
        ))

    current_y += 2

# ============================================================
#  MISE EN PAGE (LAYOUT) - AVEC THÈME APPLIQUÉ
# ============================================================

# Génération des tick labels en pourcentage pour l'axe X
x_tickvals = [i / 10 for i in range(0, 11)]          # 0.0, 0.1, ..., 1.0
x_ticktext  = [f"{int(v * 100)}%" for v in x_tickvals]  # "0%", "10%", ..., "100%"

fig.update_layout(
    title=dict(
        text="<b>Part du tremblement dans le mouvement selon l'activité</b>",
        font=dict(
            size=THEME["title_font_size"],
            color=THEME["dark_color"],
            family=THEME["font_family"]
        ),
        x=0.5,
        y=0.97
    ),
    width=FIG_WIDTH,
    height=(current_y / 2) * 70 + 150,
    paper_bgcolor=THEME["background_color"],
    plot_bgcolor=THEME["label_background_color"],
    dragmode=False,
    xaxis=dict(
        title=dict(
            text="<b>Part du tremblement dans le mouvement</b>",
            font=dict(
                size=THEME["label_font_size"],
                color=THEME["dark_color"],
                family=THEME["font_family"]
            )
        ),
        range=[-0.05, 1.05],
        tickvals=x_tickvals,
        ticktext=x_ticktext,
        tickfont=dict(
            size=THEME["label_font_size"] - 1,
            color=THEME["dark_color"],
            family=THEME["font_family"]
        ),
        gridcolor="#EEEEEE",
        zeroline=False,
        fixedrange=True
    ),
    yaxis=dict(
        tickvals=y_tick_vals,
        ticktext=y_tick_labels,
        gridcolor="#EEEEEE",
        fixedrange=True,
        tickfont=dict(
            size=THEME["label_font_size"] - 3,
            color=THEME["dark_color"],
            family=THEME["font_family"]
        )
    ),
    hoverlabel=dict(
        bgcolor=BG_TOOLTIP,
        font=dict(
            color="white",
            size=13,
            family=THEME["font_family"]
        )
    ),
    legend=dict(
        title_text="Légende :",
        title_font=dict(
            family=THEME["font_family"],
            size=THEME["label_font_size"] - 1,
            color=THEME["dark_color"]
        ),
        font=dict(
            family=THEME["font_family"],
            size=THEME["label_font_size"] - 1,
            color=THEME["dark_color"]
        ),
        bordercolor="#DDDDDD",
        borderwidth=1,
        x=1.02,
        y=1
    ),
    margin=dict(l=250, r=50, t=100, b=80),
    hovermode="closest",
    font=dict(
        family=THEME["font_family"],
        color=THEME["dark_color"]
    )
)

# ============================================================
#  EXPORT ET AFFICHAGE
# ============================================================
os.makedirs("visualizations", exist_ok=True)
fig.write_html(OUTPUT_HTML, include_plotlyjs="cdn")
fig.show()