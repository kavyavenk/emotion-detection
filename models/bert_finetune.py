from datasets import load_dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load data
dataset = load_dataset("csv", data_files={"train": "data/goemotions_train.csv", "test": "data/goemotions_test.csv"})

# Example emotion labels
emotion_labels = ['joy', 'anger', 'fear', 'sadness', 'surprise', 'disgust', 'neutral']
label2id = {label: idx for idx, label in enumerate(emotion_labels)}
id2label = {idx: label for label, idx in label2id.items()}

def encode_labels(example):
    example['label'] = label2id.get(example['emotion'], -1)
    return example

dataset = dataset.map(encode_labels)
dataset = dataset.filter(lambda x: x['label'] != -1)

# Tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize_function, batched=True)

# Model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(emotion_labels),
    id2label=id2label,
    label2id=label2id
)

# Training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1_macro": f1_score(p.label_ids, preds, average='macro')
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
