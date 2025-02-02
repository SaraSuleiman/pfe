import os
from flask import Flask, render_template, request
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 🔹 Définition des chemins des modèles
model_paths = {
    "notification": "/Users/sara/Desktop/ING5/PFE/Git/Notifications/LesModèles/notificationslegal_bert_model",
    "assignation": "/Users/sara/Desktop/ING5/PFE/Git/Notifications/LesModèles/assignationlegal_bert_model",
    "requete": "/Users/sara/Desktop/ING5/PFE/Git/Notifications/LesModèles/Requetelegal_bert_model",
    "declaration": "/Users/sara/Desktop/ING5/PFE/Git/Notifications/LesModèles/legal_bert_model"
}

# 🔹 Chargement des modèles et tokenizers
models = {}
tokenizers = {}

for key, path in model_paths.items():
    print(f"📥 Chargement du modèle : {key}")
    tokenizers[key] = AutoTokenizer.from_pretrained(path)
    models[key] = AutoModelForSequenceClassification.from_pretrained(path)
print("✅ Tous les modèles sont chargés avec succès !")

# 🔹 Dictionnaire pour mapper les classes aux messages explicites
class_mappings = {
    "notification": {
        0: "Notification : Date manquante.",
        1: "Notification : Base légale manquante.",
        2: "Notification : Référence manquante.",
        3: "Notification : Contact utile manquant.",
        4: "Notification : Signature absente.",
        5: "Notification : Aucun vice détecté.",
        6: "Notification : Aucun vice détecté."
    },
    "assignation": {
        0: "Assignation : Aucun vice détecté.",
        1: "Assignation : Absence du nom de l'avocat.",
        2: "Assignation : Absence du nom du tribunal.",
        3: "Assignation : Absence du nom de la chambre.",
        4: "Assignation : Absence de la date et l'heure de l'audience.",
        5: "Assignation : Absence de la date de l'audience.",
        6: "Assignation : Absence de l'heure de l'audience.",
        7: "Assignation : Absence de l'année.",
        8: "Assignation : Absence du nom de l'huissier.",
        9: "Assignation : Absence du nom de l'assigné.",
        10: "Assignation : Absence du lieu de naissance du demandeur.",
        11: "Assignation : Absence de la date de naissance du demandeur.",
        12: "Assignation : Absence du nom du demandeur.",
        13: "Assignation : Absence de la signature.",
        14: "Assignation : Absence de la date de l'audience."
    },
    "requete": {
        0: "Requête : Absence de la date de dépôt.",
        1: "Requête : Absence du nom de l'opposant.",
        2: "Requête : Absence du lieu de résidence de l'opposant.",
        3: "Requête : Absence de la profession du demandeur.",
        4: "Requête : Absence du lieu de résidence du demandeur.",
        5: "Requête : Aucun vice détecté.",
        6: "Requête : Absence de la signature.",
        7: "Requête : Absence du nom du tribunal.",
        8: "Requête : Absence du nom de la chambre.",
        9: "Requête : Absence du lieu de naissance du demandeur.",
        10: "Requête : Absence de la date de naissance du demandeur.",
        11: "Requête : Absence du nom du demandeur."
    },
    "declaration": {
        0: "Déclaration : Aucun vice détecté.",
        1: "Déclaration : Informations du client manquantes.",
        2: "Déclaration : Informations de l'adversaire manquantes.",
        3: "Déclaration : Informations de l'avocat manquantes.",
        4: "Déclaration : Numéro RG absent.",
        5: "Déclaration : Représentation légale manquante.",
        6: "Déclaration : Signature absente.",
        7: "Déclaration : Objet de l'appel absent.",
        8: "Déclaration : Liste des pièces manquantes."
    }
}


# 🔹 Fonction pour extraire le texte d'un PDF
def extract_text_from_pdf(file_stream):
    try:
        reader = PdfReader(file_stream)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
        return text.strip()
    except Exception as e:
        return f"Erreur lecture PDF : {e}"

# 🔹 Fonction pour effectuer la prédiction avec le bon modèle et mapping
def model_predict(text, model_key):
    if not text.strip():
        return "Le document est vide ou illisible."

    try:
        tokenizer = tokenizers[model_key]
        model = models[model_key]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1).item()

        # 🔹 Sélection du bon mapping de classes
        class_mapping = class_mappings.get(model_key, {})
        return class_mapping.get(predictions, "Nous n'arrivons pas à savoir si ce document contient un vice.")

    except Exception as e:
        return f"Erreur de prédiction : {e}"

# 🔹 Initialiser Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

# 🔹 Route pour choisir le bon modèle
@app.route('/predict/<model_key>', methods=['POST'])
def predict(model_key):
    if model_key not in models:
        return render_template("index.html", prediction="Modèle invalide.")

    pdf_file = request.files.get('pdf_file')

    if pdf_file and pdf_file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(pdf_file.stream)
        if not text:
            return render_template("index.html", prediction="Le fichier PDF est vide ou illisible.")
        prediction = model_predict(text, model_key)
        return render_template("index.html", prediction=prediction)
    else:
        return render_template("index.html", prediction="Veuillez fournir un fichier PDF valide.")

# 🔹 Lancer Flask
if __name__ == "__main__":
    app.run(port=8080)
