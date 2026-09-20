# 🛡️ Détection d'URLs Malveillantes par Machine Learning

Application de classification d'URLs (Phishing, Malware) basée sur l'extraction de caractéristiques lexicales et un modèle XGBoost, déployée avec Streamlit.

#Caractéristiques extraites
- Longueur de l'URL
- Fréquence des séparateurs et caractères suspects (`.`, `-`, `@`, `//`)
- Présence d'adresses IPv4 directes
- Entropie de Shannon (évaluation de l'aléatoire lexical)

# Installation & Exécution locale
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
