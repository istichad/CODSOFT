import re
from django.shortcuts import render
from django.http import HttpResponse
import os
from django.conf import settings
import joblib

# Charger le modèle et le vectorizer
model_path = os.path.join(settings.BASE_DIR, 'SMS', 'nb_model.pkl')
model = joblib.load(model_path)


vectorizer_path = os.path.join(settings.BASE_DIR, 'SMS', 'vectorizer.pkl')
vectorizer = joblib.load(vectorizer_path)

# Page d'accueil
def index(request):
    return render(request, 'index.html')

# Fonction pour nettoyer et prétraiter le texte
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Fonction de prédiction
def checkSMS(request):
    if request.method == "POST":
        rawData = request.POST.get("rawdata")
        if rawData:
            # Nettoyer le texte
            cleaned_data = clean_text(rawData)
            
            # Vectoriser le texte nettoyé
            vectorized_data = vectorizer.transform([cleaned_data])

            # Prédire avec le modèle
            prediction = model.predict(vectorized_data)[0]
            print(f"Prediction brute du modèle : {prediction} (type: {type(prediction)})")
            
            # Associer la prédiction à la classe correcte
            prediction_text = 'ham' if prediction.lower() == 'ham' else 'spam'
            
            # Passer la prédiction au template
            return render(request, 'output.html', {'prediction': prediction_text})
    return HttpResponse("Aucune donnée soumise.")
