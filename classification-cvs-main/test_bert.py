import torch
import joblib
from transformers import CamembertTokenizer, CamembertForSequenceClassification

print("=" * 50)
print("🧪 TEST DU MODÈLE BERT")
print("=" * 50)

# Charger le modèle et tokenizer
model = CamembertForSequenceClassification.from_pretrained("models/bert_cv_classifier")
tokenizer = CamembertTokenizer.from_pretrained("models/bert_cv_classifier")
label_mapping = joblib.load("models/bert_label_mapping.pkl")
id_to_label = {v: k for k, v in label_mapping.items()}

# Tester avec différents CVs
test_cvs = [
    "python java sql git html css javascript spring boot mysql",
    "machine learning python pandas numpy tensorflow keras",
    "cisco routing switching tcp ip network security ccna",
    "docker kubernetes aws linux jenkins terraform",
    "cybersecurity firewall penetration testing kali linux"
]

print("\n📋 Résultats des tests :")
print("-" * 50)

for cv in test_cvs:
    inputs = tokenizer(cv, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
    
    pred_id = torch.argmax(outputs.logits, dim=1).item()
    prediction = id_to_label[pred_id]
    confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()
    
    print(f"\n📄 CV: {cv[:50]}...")
    print(f"   → {prediction} (confiance: {confidence:.1%})")

print("\n" + "=" * 50)