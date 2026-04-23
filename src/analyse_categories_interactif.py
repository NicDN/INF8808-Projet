import pandas as pd
import plotly.graph_objects as go
import os

# --- 1. CONFIGURATION DU THÈME ---
THEME = {
    "background_color": "#ffffff",
    "font_family": "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif",
    "color_above_threshold": "#D55E91", # Rose
    "color_below_threshold": "#68C391", # Vert
    "threshold_line_color": "#A0AEC0",  # Gris
    "dark_color": "#2A2B2E",
    "light_text": "#718096"
}

# Chemins des fichiers
base_path = r'C:\Users\Soumeya\Desktop\data viz'
csv_path = os.path.join(base_path, 'visualizations', 'patients_symptoms.csv')
html_output = os.path.join(base_path, 'visualizations', 'analyse_categories_interactif.html')

# --- 2. TRAITEMENT DES DONNÉES ---
if not os.path.exists(csv_path):
    print(f"Erreur : Fichier introuvable : {csv_path}")
else:
    # Lecture et nettoyage des colonnes
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.replace('"', '')

    # Filtre Parkinson
    df_parkinson = df[df['condition'].str.contains('Parkinson', case=False)].copy()

    # Liste des catégories et symptômes
    categories_symptoms = {
        'Urinary': ['urgency', 'nocturia'],
        'Cognitive': ['remembering', 'loss_of_interest', 'concentrating'],
        'Sleep / Fatigue': ['daytime_sleepiness', 'insomnia', 'intense_vivid_dreams', 'acting_out_dreams', 'restless_legs'],
        'Cardiovascular': ['dizzy', 'falling', 'swelling'],
        'Gastrointestinal': ['dribbling', 'swallowing', 'vomiting', 'constipation', 'bowel_incontinence', 'bowel_emptying_incomplete'],
        'Sexual Function': ['sex_drive', 'sex_difficulty'],
        'Mood': ['sad_blues', 'anxiety'],
        'Autonomic / Systemic': ['weight_change', 'sweating'],
        'Perceptual / Psychotic': ['taste_smelling', 'hallucinations', 'diplopia', 'delusions'],
        'Pain/Somatic': ['pains']
    }

    category_data = []
    for category, symptoms in categories_symptoms.items():
        existing = [s for s in symptoms if s in df_parkinson.columns]
        if existing:
            # Calcul des moyennes
            category_percent = df_parkinson[existing].mean().mean() * 100
            
            # Choix de la couleur (Rose si > 50, sinon Vert)
            bar_color = THEME['color_above_threshold'] if category_percent > 50 else THEME['color_below_threshold']
            
            # Préparation du texte de survol (Détails)
            details_html = "".join([f" • {s.replace('_', ' ').capitalize()}: {df_parkinson[s].mean()*100:.1f}%<br>" for s in existing])
            
            hover_text = (
                f"<span style='font-size:16px; color:{bar_color};'><b>{category.upper()}</b></span><br>"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
                f"Moyenne globale : <b>{category_percent:.1f}%</b><br><br>"
                f"<b>Détails :</b><br>"
                f"<span style='font-family:monospace; font-size:12px;'>{details_html}</span>"
            )

            category_data.append({
                'Catégorie': category,
                'Pourcentage': category_percent,
                'Couleur': bar_color,
                'Hover': hover_text
            })

    # Création du tableau trié
    cat_df = pd.DataFrame(category_data).sort_values('Pourcentage', ascending=False)

    # --- 3. CRÉATION DU GRAPHIQUE ---
    fig = go.Figure()

    # Ajout des barres
    fig.add_trace(go.Bar(
        x=cat_df['Catégorie'],
        y=cat_df['Pourcentage'],
        marker=dict(color=cat_df['Couleur']), 
        hovertext=cat_df['Hover'],
        hoverinfo='text',
        text=[f"<b>{v:.1f}%</b>" for v in cat_df['Pourcentage']],
        textposition='outside',
        cliponaxis=False
    ))

    # Ligne de seuil à 50%
    fig.add_hline(
        y=50, 
        line_dash="dash", 
        line_color=THEME['threshold_line_color'], 
        line_width=2,
        annotation_text="Seuil d'alerte (50%)", 
        annotation_position="top right",
        annotation_font=dict(size=12, color=THEME['light_text'])
    )

    # --- 4. MISE EN FORME FINALE ---
    fig.update_layout(
        font_family=THEME['font_family'],
        # Titre centré
        title=dict(
            text="<b>Prévalence des Catégories de Symptômes Non Moteurs</b><br><span style='font-size:14px; color:#718096;'>Moyenne par groupe de symptômes (Parkinson)</span>",
            x=0.5,
            xanchor='center',
            font=dict(size=22, color=THEME['dark_color'])
        ),
        # Axe X
        xaxis=dict(
            title=dict(text="<b>Catégories de Symptômes Non Moteurs</b>", standoff=30),
            tickangle=-45,
            showgrid=False
        ),
        # Axe Y
        yaxis=dict(
            title=dict(text="<b>Pourcentage moyen (%)</b>", standoff=10),
            range=[0, 105],
            gridcolor='#EDF2F7',
            zeroline=False
        ),
        template="plotly_white",
        height=700,
        margin=dict(l=80, r=80, t=120, b=150),
        paper_bgcolor="white",
        plot_bgcolor="white",
        
        # Style de la fenêtre de survol
        hoverlabel=dict(
            bgcolor="white",
            font_size=13,
            font_color=THEME['dark_color'],
            bordercolor="#E2E8F0",
            align="left"
        )
    )

    # Export HTML
    fig.write_html(html_output, include_plotlyjs='cdn')
    print(f"Graphique final généré avec succès : {html_output}")
