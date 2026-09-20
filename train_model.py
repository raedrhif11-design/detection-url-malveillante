import math
import os
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def calculate_shannon_entropy(text: str) -> float:
    """Calcule l'entropie de Shannon pour évaluer l'aléatoire lexical d'une chaîne."""
    if not text:
        return 0.0
    probabilities = [float(text.count(c)) / len(text) for c in dict.fromkeys(list(text))]
    return -sum([p * math.log(p) / math.log(2.0) for p in probabilities])

def extract_features(url: str) -> list:
    """Extrait les métriques lexicales et structurelles d'une URL."""
    features = [
        len(url),                                      # Longueur totale
        url.count('.'),                                # Nombre de points
        url.count('-'),                                # Nombre de tirets
        url.count('@'),                                # Présence d'arrobase (phishing classique)
        url.count('?'),                                # Paramètres de requête
        url.count('='),                                # Affectations
        url.count('//'),                               # Présence de doubles slashes
        int(bool(re.search(r'\d{1,3}(\.\d{1,3}){3}', url))), # Présence d'adresse IPv4
        sum(c.isdigit() for c in url),                 # Nombre de chiffres
        calculate_shannon_entropy(url)                 # Entropie de Shannon
    ]
    return features

def main():
    print("[*] Génération du dataset d'entraînement...")
    sample_data = {
        'url': [
            'google.com', 'wikipedia.org', 'github.com', 'stackoverflow.com',
            'insa-hautsdefrance.fr', 'lemonde.fr', 'amazon.com', 'microsoft.com',
            'http://192.168.1.1/login-verify-account.html',
            'http://secure-paypal-update-alert.com/login@verify',
            'http://bit.ly/malicious-login-credential-stealer',
            'http://free-crypto-giveaway-claim-now.xyz/signin',
            'http://apple-id-suspended-security-check.com',
            'http://10.0.0.5//session/token?id=99281&action=exec'
        ] * 100,
        'label': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1] * 100
    }
    df = pd.DataFrame(sample_data)

    print("[*] Extraction des caractéristiques...")
    X = np.array([extract_features(u) for u in df['url']])
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("[*] Entraînement du modèle XGBoost...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[+] Précision sur le jeu de test : {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'xgboost_url_model.pkl')
    print("[+] Modèle sauvegardé sous : xgboost_url_model.pkl")

if __name__ == '__main__':
    main()
