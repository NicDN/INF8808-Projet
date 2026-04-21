import pandas as pd
import plotly.graph_objects as go
import os

# --- 1. CONFIGURATION DU THÈME (Texte, Tailles, Fond) ---
# On centralise ici le style pour avoir un graphique moderne et uniforme
THEME = {
    "background_color": "#ffffff",
    "font_family": "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif",
    "accent_font_family": "Inter, 'Segoe UI', system-ui, sans-serif",
    "dark_color": "#2A2B2E",
    "label_font_size": 14,
    "title_font_size": 20,
    "label_background_color": "#ffffff",
}

# Chemins des fichiers
base_path = r'C:\Users\Soumeya\Desktop\data viz'
csv_path = os.path.join(base_path, 'visualizations', 'patients_symptoms.csv')
html_output = os.path.join(base_path, 'visualizations', 'analyse_categories_interactif.html')

# --- 2. TRAITEMENT DES DONNÉES ---
# Lecture du csv et petit nettoyage des colonnes (espaces/guillemets)
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.replace('"', '')

# On garde uniquement les patients qui ont Parkinson
df_parkinson = df[df['condition'].str.contains('Parkinson', case=False)].copy()

# Regroupement des symptômes par grandes catégories
categories_symptoms = {
    'Gastrointestinal': ['dribbling', 'swallowing', 'vomiting', 'constipation', 
                         'bowel_incontinence', 'bowel_emptying_incomplete'],
    'Urinary': ['urgency', 'nocturia'],
    'Pain/Somatic': ['pains'],
    'Autonomic / Systemic': ['weight_change', 'sweating'],
    'Cognitive': ['remembering', 'loss_of_interest', 'concentrating'],
    'Perceptual / Psychotic': ['taste_smelling', 'hallucinations', 'diplopia', 'delusions'],
    'Mood': ['sad_blues', 'anxiety'],
    'Sexual Function': ['sex_drive', 'sex_difficulty'],
    'Cardiovascular': ['dizzy', 'falling', 'swelling'],
    'Sleep / Fatigue': ['daytime_sleepiness', 'insomnia', 'intense_vivid_dreams', 
                        'acting_out_dreams', 'restless_legs']
}

# Couleurs originales pour les barres
colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', 
          '#E6AB02', '#A6761D', '#666666', '#1F78B4', '#FF7F00']

# Boucle pour calculer les moyennes par catégorie et préparer les tooltips
category_data = []
for i, (category, symptoms) in enumerate(categories_symptoms.items()):
    existing = [s for s in symptoms if s in df_parkinson.columns]
    if existing:
        # Calcul de la moyenne globale pour la catégorie
        category_percent = df_parkinson[existing].mean().mean() * 100
        
        # On prépare le texte qui s'affiche au survol (HTML pour Plotly)
        details_html = ""
        for sym in existing:
            s_val = df_parkinson[sym].mean() * 100
            details_html += f" • {sym.replace('_', ' ').capitalize():<22} : {s_val:>5.1f}%<br>"
        
        # Message si ça dépasse 50%
        status = "<b>✓ DÉPASSE LE SEUIL</b>" if category_percent > 50 else "✗ SOUS LE SEUIL"
        color_status = "#E53E3E" if category_percent > 50 else "#718096"

        hover_text = (
            f"<span style='font-family:{THEME['accent_font_family']}; font-size:16px; font-weight:bold; color:{colors[i]};'>"
            f"CATÉGORIE : {category.upper()}</span><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"Moyenne : <b>{category_percent:.1f}%</b> des patients<br>"
            f"<span style='color:{color_status};'>{status}</span><br>"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
            f"Détails par symptôme :<br>"
            f"<span style='font-family:monospace; font-size:12px;'>{details_html}</span>"
        )

        category_data.append({
            'Catégorie': category,
            'Pourcentage': category_percent,
            'Hover': hover_text,
            'Couleur Originale': colors[i]
        })

# Tri des résultats du plus grand au plus petit
cat_df = pd.DataFrame(category_data).sort_values('Pourcentage', ascending=False)

# --- 3. CRÉATION DU GRAPHIQUE ---
fig = go.Figure()

fig.add_trace(go.Bar(
    x=cat_df['Catégorie'],
    y=cat_df['Pourcentage'],
    marker=dict(color=cat_df['Couleur Originale']), 
    hovertext=cat_df['Hover'],
    hoverinfo='text',
    text=[f"<b>{v:.1f}%</b>" for v in cat_df['Pourcentage']],
    textposition='outside'
))

# Ajout de la ligne rouge pour le seuil de 50%
fig.add_hline(y=50, line_dash="dash", line_color="#E74C3C", line_width=2,
              annotation_text="Seuil d'alerte (50%)", annotation_position="top right")

# --- 4. MISE EN FORME DU GRAPHIQUE (titres, polices, couleurs de fond) ---
fig.update_layout(
    font_family=THEME['font_family'],
    title=dict(
        text="<b>Prévalence des Catégories de Symptômes Non Moteurs</b><br>"
             "<span style='font-size:14px; color:#718096;'>Moyenne par groupe de symptômes (Parkinson)</span>",
        x=0.5, 
        font=dict(size=THEME['title_font_size'], color=THEME['dark_color'])
    ),
    xaxis=dict(
        # On garde le titre original de l'axe X
        title=dict(
            text="<b>Catégories de Symptômes Non Moteurs</b>", 
            font=dict(size=THEME['label_font_size'])
        ),
        tickangle=-45
    ),
    yaxis=dict(
        title=dict(
            text="<b>Pourcentage moyen (%)</b>", 
            font=dict(size=THEME['label_font_size'])
        ),
        range=[0, 105],
        gridcolor='#EDF2F7'
    ),
    template="plotly_white",
    height=700,
    margin=dict(l=80, r=80, t=120, b=150),
    paper_bgcolor=THEME['background_color'],
    plot_bgcolor=THEME['background_color'],
    
    # Style des tooltips (hover)
    hoverlabel=dict(
        bgcolor=THEME['label_background_color'],
        font_size=13,
        font_family=THEME['font_family'],
        align="left"
    )
)

# Export en fichier HTML autonome
fig.write_html(html_output, include_plotlyjs='cdn')
print(f"Graphique mis à jour avec le style Inter et tes couleurs : {html_output}")
