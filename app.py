import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Charger le modèle et le tokenizer fine-tuné
model_path = "./legal_bert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Dictionnaire pour mapper les classes aux messages explicites
class_mapping = {
    0: "Assignation : Ce document présente un vice, car le nom du demandeur est absent.",
    1: "Assignation : Ce document présente un vice, car le nom du tribunal est absent.",
    2: "Requête : Ce document présente un vice, car le nom de l'opposant est absent.",
    3: "Requête : Ce document présente un vice, car la date de naissance du demandeur est absente.",
    4: "Requête : Ce document présente un vice, car le nom du demandeur est absent.",
    5: "Assignation : Ce document présente un vice, car le lieu de naissance du demandeur est absent.",
    6: "Requête : Ce document présente un vice, car le lieu de naissance du demandeur est absent.",
    7: "Assignation : Ce document ne présente aucun vice.",
    8: "Requête : Ce document présente un vice, car la date de dépôt est absente.",
    9: "Requête : Ce document présente un vice, car le nom de la chambre est absent.",
    10: "Assignation : Ce document présente un vice, car le nom de l'huissier est absent.",
    11: "Assignation : Ce document présente un vice, car le nom de la chambre est absent.",
    12: "Assignation : Ce document présente un vice, car le nom de l'avocat est absent.",
    13: "Requête : Ce document présente un vice, car le lieu de résidence du demandeur est absent.",
    14: "Assignation : Ce document présente un vice, car la date et l’heure de l’audience sont absentes.",
    15: "Assignation : Ce document présente un vice, car la signature est absente.",
    16: "Requête : Ce document présente un vice, car la profession du demandeur est absente.",
    17: "Assignation : Ce document présente un vice, car le nom de l'assigné est absent.",
    18: "Assignation : Ce document présente un vice, car la date de naissance du demandeur est absente.",
    19: "Assignation : Ce document présente un vice, car l’année est absente.",
    20: "Requête : Ce document présente un vice, car le nom du tribunal est absent.",
    21: "Requête : Ce document présente un vice, car le lieu de résidence de l'opposant est absent.",
    22: "Assignation : Ce document présente un vice, car la date de l’audience est absente.",
    23: "Requête : Ce document présente un vice, car la signature est absente.",
    24: "Assignation : Ce document présente un vice, car l’heure de l’audience est absente.",
    25: "Requête : Ce document ne présente aucun vice.",
    26: "Assignation : Ce document présente un vice, car la date de l’audience est absente."
}


# Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(file_stream):
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Erreur lors de la lecture du PDF : {e}")
        return ""

# Fonction pour effectuer la prédiction
def model_predict(text):
    if not text.strip():
        return "Le document est vide ou illisible."

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).item()
        return class_mapping.get(predictions, "Classe inconnue")
    except Exception as e:
        print(f"Erreur lors de la prédiction : {e}")
        return "Erreur lors de la prédiction."

# Initialiser Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    pdf_file = request.files.get('pdf_file')

    if pdf_file and pdf_file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(pdf_file.stream)
        if not text:
            return render_template("index.html", prediction="Le fichier PDF est vide ou illisible.")
        prediction = model_predict(text)
        return render_template("index.html", prediction=prediction)
    else:
        return render_template("index.html", prediction="Veuillez fournir un fichier PDF valide.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Utiliser le port $PORT ou 8080 par défaut
    app.run(host="0.0.0.0", port=port, debug=False)
