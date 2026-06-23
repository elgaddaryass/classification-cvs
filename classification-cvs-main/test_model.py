import joblib
import torch
from transformers import CamembertTokenizer, CamembertForSequenceClassification

# Test Naive Bayes
print("=" * 50)
print("🧪 Test des deux modèles")
print("=" * 50)

# 1. Naive Bayes
nb_model = joblib.load("models/cv_classifier.pkl")
test_cv = "python java sql git html css javascript spring boot"
pred_nb = nb_model.predict([test_cv])[0]
print(f"\n📌 Naive Bayes:")
print(f"   CV: {test_cv}")
print(f"   Prédiction: {pred_nb}")

# 2. BERT
bert_model = CamembertForSequenceClassification.from_pretrained("models/bert_cv_classifier")
tokenizer = CamembertTokenizer.from_pretrained("models/bert_cv_classifier")
label_mapping = joblib.load("models/bert_label_mapping.pkl")
id_to_label = {v: k for k, v in label_mapping.items()}

inputs = tokenizer(test_cv, return_tensors="pt", truncation=True, max_length=256)
with torch.no_grad():
    outputs = bert_model(**inputs)
pred_id = torch.argmax(outputs.logits, dim=1).item()
pred_bert = id_to_label[pred_id]
confidence = torch.softmax(outputs.logits, dim=1)[0][pred_id].item()

print(f"\n🤖 BERT:")
print(f"   CV: {test_cv}")
print(f"   Prédiction: {pred_bert}")
print(f"   Confiance: {confidence:.2%}")