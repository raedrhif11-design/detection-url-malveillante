import streamlit as st
import numpy as np
import joblib
import math
import re
import os

st.set_page_config(page_title="Détecteur d'URLs Malveillantes", page_icon="🛡️", layout="centered")

def calculate_shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    probabilities = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum([p * math.log(p) / math.log(2.0) for p in probabilities])

def extract_features(url: str) -> list:
    return [
        len(url),
        url.count('.'),
        url.count('-'),
        url.count('@'),
        url.count('?'),
        url.count('='),
        url.count('//'),
        int(bool(re.search(r'\d{1,3}(\.\d{1,3}){3}', url))),
        sum(c.isdigit() for c in url),
        calculate_shannon_entropy(url)
    ]

@st.cache_resource
def load_model():
    if os.path.exists('xgboost_url_model.pkl'):
        return joblib.load('xgboost_url_model.pkl')
    return None

st.title("🛡️ Détection d'URLs Malveillantes (ML)")
st.write("Analyse lexicale et classification en temps réel via un modèle **XGBoost**.")

model = load_model()

url_input = st.text_input("Entrez une URL à analyser :", placeholder="ex: https://service-banque-verification.com/login")

if st.button("Analyser l'URL"):
    if not url_input.strip():
        st.warning("Veuillez saisir une URL valide.")
    elif model is None:
        st.error("Modèle introuvable. Exécutez d'abord train_model.py pour générer xgboost_url_model.pkl.")
    else:
        features = np.array([extract_features(url_input)])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        st.subheader("Résultats du diagnostic")
        if prediction == 1:
            st.error(f"🚨 **Alerte : URL Malveillante détectée** (Risque : {probabilities[1]*100:.1f}%)")
        else:
            st.success(f"✅ **URL Légitime** (Indice de confiance : {probabilities[0]*100:.1f}%)")

        with st.expander("Détail des métriques extraites"):
            st.json({
                "Longueur": len(url_input),
                "Entropie de Shannon": round(calculate_shannon_entropy(url_input), 3),
                "Nombre de chiffres": sum(c.isdigit() for c in url_input),
                "Présence IP": bool(re.search(r'\d{1,3}(\.\d{1,3}){3}', url_input))
            })
