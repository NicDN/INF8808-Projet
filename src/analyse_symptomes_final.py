import pandas as pd
import plotly.graph_objects as go
import os

# --- Setup des dossiers ---
base_path = r'C:\Users\Soumeya\Desktop\data viz'
csv_path = os.path.join(base_path, 'visualizations', 'patients_symptoms.csv')
html_output = os.path.join(base_path, 'visualizations', 'analyse_symptomes_final.html')

# Chargement des données et petit clean sur les noms de colonnes
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.replace('"', '')

# On filtre pour ne garder que les cas de Parkinson
df_parkinson = df[df['condition'].str.contains('Parkinson', case=False)].copy()

# Dico pour faire le lien entre les colonnes et les noms propres
detailed_info = {
    'dribbling': 'Dribbling', 'swallowing': 'Swallowing', 'vomiting': 'Vomiting',
    'constipation': 'Constipation', 'bowel_incontinence': 'Bowel Incontinence',
    'bowel_emptying_incomplete': 'Bowel Emptying Incomplete', 'urgency': 'Urgency',
    'nocturia': 'Nocturia', 'pains': 'Pains', 'weight_change': 'Weight Change',
    'sweating': 'Sweating', 'remembering': 'Remembering', 'loss_of_interest': 'Loss of Interest',
    'concentrating': 'Concentrating', 'taste_smelling': 'Taste/Smelling',
    'hallucinations': 'Hallucinations', 'diplopia': 'Diplopia', 'delusions': 'Delusions',
    'sad_blues': 'Sad Blues', 'anxiety': 'Anxiety', 'sex_drive': 'Sex Drive',
    'sex_difficulty': 'Sex Difficulty', 'dizzy': 'Dizzy', 'falling': 'Falling',
    'swelling': 'Swelling', 'daytime_sleepiness': 'Daytime Sleepiness',
    'insomnia': 'Insomnia', 'intense_vivid_dreams': 'Intense Vivid Dreams',
    'acting_out_dreams': 'Acting Out Dreams', 'restless_legs': 'Restless Legs'
}

# Liste des symptômes par catégorie pour la boucle
sub_symptoms = {
    'Gastrointestinal': ['dribbling', 'swallowing', 'vomiting', 'constipation', 'bowel_incontinence', 'bowel_emptying_incomplete'],
    'Urinary': ['urgency', 'nocturia'],
    'Pain/Somatic': ['pains'],
    'Autonomic / Systemic': ['weight_change', 'sweating'],
    'Cognitive': ['remembering', 'loss_of_interest', 'concentrating'],
    'Perceptual / Psychotic': ['taste_smelling', 'hallucinations', 'diplopia', 'delusions'],
    'Mood': ['sad_blues', 'anxiety'],
    'Sexual Function': ['sex_drive', 'sex_difficulty'],
    'Cardiovascular': ['dizzy', 'falling', 'swelling'],
    'Sleep / Fatigue': ['daytime_sleepiness', 'insomnia', 'intense_vivid_dreams', 'acting_out_dreams', 'restless_legs']
}

# Calcul des stats et préparation des tooltips (hover)
sub_data = []
for category, symptoms in sub_symptoms.items():
    for symptom in symptoms:
        if symptom in df_parkinson.columns:
            # Pourcentage global
            percent = df_parkinson[symptom].mean() * 100
            
            # Calcul de la répartition Homme/Femme pour ceux qui ont le symptôme
            pts_with_symptom = df_parkinson[df_parkinson[symptom] == 1]
            total_with_symptom = len(pts_with_symptom)
            
            if total_with_symptom > 0:
                males = pts_with_symptom[pts_with_symptom['sex'].str.strip().str.lower() == 'male']
                females = pts_with_symptom[pts_with_symptom['sex'].str.strip().str.lower() == 'female']
                p_hommes = (len(males) / total_with_symptom) * 100
                p_femmes = (len(females) / total_with_symptom) * 100
            else:
                p_hommes, p_femmes = 0.0, 0.0

            # Texte personnalisé pour le survol de la souris
            hover_text = (
                f"<span style='font-size:15px; font-weight:bold; color:#276749;'>"
                f"{detailed_info.get(symptom, symptom).upper()}</span><br>"
                f"<span style='color:#718096;'>Catégorie : {category}</span><br><br>"
                f"<b>Fréquence :</b> {percent:.1f}%<br>"
                f"<span style='font-size:11px; color:#A0AEC0;'>({total_with_symptom}/{len(df_parkinson)} patients)</span><br><br>"
                f"<b>Genre :</b><br>"
                f" <span style='color:#3182CE;'>• Hommes : {p_hommes:.1f}%</span><br>"
                f" <span style='color:#E53E3E;'>• Femmes : {p_femmes:.1f}%</span>"
            )

            sub_data.append({
                'Nom': detailed_info.get(symptom, symptom),
                'Pourcentage': percent,
                'Hover': hover_text
            })

# Tri pour avoir les plus fréquents en haut
sub_df = pd.DataFrame(sub_data).sort_values('Pourcentage', ascending=True)

# Création du bar chart horizontal avec Plotly
fig = go.Figure()

fig.add_trace(go.Bar(
    y=sub_df['Nom'],
    x=sub_df['Pourcentage'],
    orientation='h',
    marker=dict(color='#48BB78', line=dict(color='#FFFFFF', width=1.5)), # Vert menthe
    hovertext=sub_df['Hover'],
    hoverinfo='text',
    text=[f"<b>{v:.1f}%</b>" for v in sub_df['Pourcentage']],
    textposition='outside'
))

# Réglages du design (titres, axes, couleurs de fond)
fig.update_layout(
    title=dict(
        text="<b>Répartition des symptômes non moteurs (en % des patients)</b>",
        x=0.5, font=dict(size=22, color='#2D3748')
    ),
    xaxis=dict(title="Pourcentage des patients (%)", range=[0, 100], gridcolor='#EDF2F7'),
    
    # Label de l'axe Y
    yaxis=dict(
        title="<b>Sous-Symptômes</b>", 
        title_font=dict(size=14, color='#2D3748'),
        tickfont=dict(size=11),
        automargin=True
    ),
    
    template="plotly_white",
    height=950,
    margin=dict(l=220, r=80, t=100, b=80),
    paper_bgcolor='#FFFDFB', # Fond blanc cassé
    plot_bgcolor='#FFFDFB',
    
    # Style de la petite fenêtre au survol
    hoverlabel=dict(
        bgcolor='#F0FFF4',
        bordercolor='#48BB78',
        font_size=13,
        font_color="#2D3748",
        align="left"
    )
)

# Export final en HTML pour le mettre sur Git
fig.write_html(html_output, include_plotlyjs='cdn')
print(f"Graphique généré avec succès dans : {html_output}")