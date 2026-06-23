###* l’utilisateur interagit avec l’application.

import os
import pandas as pd
import joblib
import streamlit as st

from utils import extract_text_from_pdf, clean_text
from agent import CVAgent


# Créer les dossiers nécessaires s'ils n'existent pas
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)


# Chemin du modèle entraîné
MODEL_PATH = "models/cv_classifier.pkl"


# Vérifier si le modèle existe
if not os.path.exists(MODEL_PATH):
    st.error("Modèle introuvable. Lance d'abord cette commande : python train_model.py")
    st.stop()


# Charger le modèle
model = joblib.load(MODEL_PATH)


# Créer l'agent IA
agent = CVAgent(model)


# Interface Streamlit
st.title("Agent IA pour l'analyse et le classement des CVs")

st.write(
    "Cette application reçoit un profil recherché, analyse plusieurs CVs PDF, "
    "puis classe les candidats automatiquement selon leur priorité."
)


# Zone où l'utilisateur écrit le profil recherché
profile_text = st.text_area(
    "Profil recherché",
    placeholder="Exemple : Développeur backend avec Python, Django, SQL, Docker, Git et REST API.",
    height=150
)


# Upload de plusieurs CVs
uploaded_files = st.file_uploader(
    "Uploader plusieurs CVs PDF",
    type=["pdf"],
    accept_multiple_files=True
)


# Bouton de lancement
if st.button("Analyser et classer les CVs"):

    # Vérifier que le profil est saisi
    if profile_text.strip() == "":
        st.error("Veuillez saisir le profil recherché.")
        st.stop()

    # Vérifier qu'au moins un CV est envoyé
    if uploaded_files is None or len(uploaded_files) == 0:
        st.error("Veuillez uploader au moins un CV.")
        st.stop()

    # Nettoyer le profil recherché
    cleaned_profile = clean_text(profile_text)

    cvs = []

    # Lire chaque CV envoyé
    for uploaded_file in uploaded_files:

        # Sauvegarder le CV dans le dossier uploads
        file_path = os.path.join("uploads", uploaded_file.name)

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        # Extraire le texte du PDF
        extracted_text = extract_text_from_pdf(file_path)

        # Si le CV est vide ou scanné
        if extracted_text.strip() == "":
            st.warning(f"Aucun texte extrait depuis : {uploaded_file.name}")
            continue

        # Nettoyer le texte extrait
        cleaned_cv_text = clean_text(extracted_text)

        # Ajouter le CV à la liste
        cvs.append({
            "name": uploaded_file.name,
            "text": cleaned_cv_text
        })

    # Vérifier qu'il y a au moins un CV exploitable
    if len(cvs) == 0:
        st.error("Aucun CV exploitable. Les CVs sont peut-être scannés ou vides.")
        st.stop()

    # Appeler l'agent pour classer les CVs
    results = agent.rank_cvs(cleaned_profile, cvs)

    # Afficher les résultats
    st.subheader("Classement des CVs")

    table_data = []

    for index, result in enumerate(results, start=1):
      table_data.append({
         "Rang": index,
         "CV": result["cv_name"],
         "Catégorie": result["cv_category"],
         "Score": result["final_score"],
         "Priorité": result["priority"]
      })

    df_results = pd.DataFrame(table_data)

    st.table(df_results)

    # Afficher le meilleur candidat
    st.subheader("Meilleur candidat")

    best = results[0]

    st.success(f"Meilleur CV : {best['cv_name']}")
    st.write("Score final :", best["final_score"])
    st.write("Priorité :", best["priority"])
    st.write("Catégorie prédite :", best["cv_category"])

    if len(best["cv_skills"]) > 0:
        st.write("Compétences détectées :", ", ".join(best["cv_skills"]))
    else:
        st.write("Compétences détectées : aucune compétence détectée")

    # Détails de chaque CV
    st.subheader("Détails par candidat")

    for result in results:
        with st.expander(
            f"{result['cv_name']} — {result['priority']} — Score : {result['final_score']}"
        ):
            st.write("Catégorie du profil recherché :", result["profile_category"])
            st.write("Catégorie du CV :", result["cv_category"])
            st.write("Similarité texte :", str(result["similarity_score"]) + "%")
            st.write("Correspondance compétences :", str(result["skills_score"]) + "%")

            if len(result["profile_skills"]) > 0:
                st.write("Compétences demandées :", ", ".join(result["profile_skills"]))
            else:
                st.write("Compétences demandées : aucune compétence détectée dans le profil")

            if len(result["cv_skills"]) > 0:
                st.write("Compétences du CV :", ", ".join(result["cv_skills"]))
            else:
                st.write("Compétences du CV : aucune compétence détectée")