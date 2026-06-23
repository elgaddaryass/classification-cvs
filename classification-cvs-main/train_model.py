# comment le modèle apprend.


import pandas as pd
import joblib
import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from utils import clean_text


# 1. Lire le dataset
data = pd.read_csv("data/cvs.csv")

# 2. Vérifier les colonnes
if "text" not in data.columns or "label" not in data.columns:
    raise ValueError("Le fichier cvs.csv doit contenir les colonnes: text,label")

# 3. Nettoyer les textes
data["text"] = data["text"].apply(clean_text)

# 4. Séparer texte et catégorie
X = data["text"]
y = data["label"]

# 5. Diviser les données : 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Créer le modèle
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", MultinomialNB())
])

# 7. Entraîner sur 80%
model.fit(X_train, y_train)

# 8. Tester sur 20%
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))

# 9. Créer le modèle final entraîné sur 100% des données
final_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000
    )),
    ("classifier", MultinomialNB())
])

final_model.fit(X, y)

# 10. Sauvegarder le modèle final
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/cv_classifier.pkl")

print("Final model trained on all data and saved successfully.")