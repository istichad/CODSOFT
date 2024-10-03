import os
from django.shortcuts import render
from django.conf import settings
import joblib
import pandas as pd

# Charger le modèle et le scaler
model_path = os.path.join(settings.BASE_DIR, 'bank', 'model_rf.pkl')
model = joblib.load(model_path)

scaler_path = os.path.join(settings.BASE_DIR, 'bank', 'scaler.pkl')
scaler = joblib.load(scaler_path)

# Page d'accueil
def index(request):
    return render(request, 'index.html')

# Fonction de prédiction
def predictClient(request):
    if request.method == "POST":
        # Récupérer les données du formulaire
        input_data = {
            'CreditScore': float(request.POST['CreditScore']),
            'Age': float(request.POST['Age']),
            'Tenure': int(request.POST['Tenure']),
            'Balance': float(request.POST['Balance']),
            'NumOfProducts': int(request.POST['NumOfProducts']),
            'HasCrCard': 1 if request.POST['HasCrCard'] == 'Yes' else 0,
            'IsActiveMember': 1 if request.POST['IsActiveMember'] == 'Yes' else 0,
            'EstimatedSalary': float(request.POST['EstimatedSalary']),
            'Geography': request.POST['Geography'],
            'Gender': request.POST['Gender']
        }

        # Encodage des colonnes Geography
        input_data['Geography_Germany'] = 0
        input_data['Geography_Spain'] = 0
        if input_data['Geography'] == 'Germany':
            input_data['Geography_Germany'] = 1
        elif input_data['Geography'] == 'Spain':
            input_data['Geography_Spain'] = 1

        # Encodage des colonnes Gender
        input_data['Gender_Male'] = 1 if input_data['Gender'] == 'Male' else 0

        # Supprimer les colonnes non encodées
        del input_data['Geography']
        del input_data['Gender']

        # Conversion des données en DataFrame
        df_input = pd.DataFrame([input_data])

        # Standardisation des variables continues
        columns_to_scale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
        df_input[columns_to_scale] = scaler.transform(df_input[columns_to_scale])

        # Prédiction avec le modèle
        prediction = model.predict(df_input)

        # Mappage des résultats de la prédiction (binaire -> catégorie)
        result_message = "Le client va quitter." if prediction[0] == 1 else "Le client va rester."

        # Retourner la page de résultat avec le message
        return render(request, 'output.html', {'result_message': result_message})
    
    return render(request, 'index.html')
