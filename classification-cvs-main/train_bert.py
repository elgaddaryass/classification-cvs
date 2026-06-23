import pandas as pd
import torch
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import CamembertTokenizer, CamembertForSequenceClassification
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("📂 Chargement des données...")
data = pd.read_csv("data/cvs.csv")

def simple_clean(text):
    return str(text).lower().strip()

data["text"] = data["text"].apply(simple_clean)

labels_list = data["label"].unique().tolist()
label_to_id = {label: idx for idx, label in enumerate(labels_list)}
id_to_label = {idx: label for label, idx in label_to_id.items()}
data["label_id"] = data["label"].map(label_to_id)

print(f"📊 Classes: {labels_list}")
print(f"📈 Distribution:\n{data['label'].value_counts()}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label_id"], test_size=0.2, random_state=42, stratify=data["label_id"]
)

print(f"🔹 Train: {len(X_train)} CVs | Test: {len(X_test)} CVs")

# Tokenisation
print("🔄 Tokenisation...")
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-base")

def tokenize_function(texts):
    return tokenizer(
        texts.tolist(),
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

train_encodings = tokenize_function(X_train)
test_encodings = tokenize_function(X_test)

class CVDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = CVDataset(train_encodings, y_train)
test_dataset = CVDataset(test_encodings, y_test)

# Modèle
print("🤖 Chargement du modèle...")
model = CamembertForSequenceClassification.from_pretrained(
    "camembert/camembert-base",
    num_labels=len(labels_list),
    ignore_mismatched_sizes=True
)

# Version CORRIGÉE des TrainingArguments
training_args = TrainingArguments(
    output_dir="./bert_results",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=5,
    weight_decay=0.01,
    logging_dir="./bert_logs",
    logging_steps=10,
    eval_strategy="epoch",  # 👈 Changé: 'evaluation_strategy' → 'eval_strategy'
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": (predictions == labels).mean()}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("🚀 Début de l'entraînement...")
trainer.train()

print("\n📊 Évaluation:")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=1)
print(classification_report(y_test, preds, target_names=labels_list))

print("💾 Sauvegarde...")
model.save_pretrained("models/bert_cv_classifier")
tokenizer.save_pretrained("models/bert_cv_classifier")
joblib.dump(label_to_id, "models/bert_label_mapping.pkl")
joblib.dump(id_to_label, "models/bert_id_to_label.pkl")

print("\n✅ Modèle BERT sauvegardé avec succès!")