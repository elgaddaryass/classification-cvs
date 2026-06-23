# Comparez Naive Bayes vs BERT


import joblib
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification

print("=" * 60)
print("📊 COMPARAISON FINALE : Naive Bayes vs BERT")
print("=" * 60)

# Charger Naive Bayes
nb_model = joblib.load("models/cv_classifier.pkl")

# Charger BERT
bert_model = CamembertForSequenceClassification.from_pretrained("models/bert_cv_classifier")
tokenizer = CamembertTokenizer.from_pretrained("models/bert_cv_classifier")
label_mapping = joblib.load("models/bert_label_mapping.pkl")
id_to_label = {v: k for k, v in label_mapping.items()}

# CVs de test
test_cvs = [
    ("python java react nodejs mongodb express", "Software Engineering"),
    ("pandas numpy scikit-learn tensorflow data science", "Data Science"),
    ("router switch cisco vlan tcp ip firewall", "Networks"),
    ("kali linux metasploit burp suite security audit", "Cybersecurity"),
    ("docker kubernetes aws terraform jenkins ci cd", "Cloud DevOps"),
]

print("\n📋 Résultats détaillés :")
print("-" * 60)

for cv, expected in test_cvs:
    print(f"\n🔹 CV: {cv[:40]}...")
    print(f"   Attendu: {expected}")
    
    # Naive Bayes
    pred_nb = nb_model.predict([cv])[0]
    
    # BERT
    inputs = tokenizer(cv, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    pred_bert = id_to_label[pred_id]
    confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()
    
    nb_correct = "✅" if pred_nb == expected else "❌"
    bert_correct = "✅" if pred_bert == expected else "❌"
    
    print(f"   🔵 Naive Bayes: {pred_nb} {nb_correct}")
    print(f"   🟢 BERT:        {pred_bert} (conf:{confidence:.0%}) {bert_correct}")

print("\n" + "=" * 60)