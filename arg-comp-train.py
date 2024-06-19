# Step 1: Install the necessary libraries (if not already installed)
# pip install transformers datasets evaluate

# Step 2: Import the necessary libraries
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate
import os

# Import custom functions from util.py
from utils import read_conll_file, create_hf_dataset, tokenize_and_align_labels, compute_metrics
import config

configuration = config.Config("abstrct", "allenai/scibert_scivocab_uncased", "allenai/scibert_scivocab_uncased", "scibert")

# Create the directory to store the model
os.makedirs(f"./models/{configuration.name}/{configuration.model_name}", exist_ok=True)

# Step 3: Read the CoNLL files
train_sentences, train_labels = read_conll_file(f"datasets/{configuration.name}/train.conll")
dev_sentences, dev_labels = read_conll_file(f"datasets/{configuration.name}/dev.conll")
test_sentences, test_labels = read_conll_file(f"datasets/{configuration.name}/test.conll")

# Step 4: Create Hugging Face datasets
train_dataset = create_hf_dataset(train_sentences, train_labels)
dev_dataset = create_hf_dataset(dev_sentences, dev_labels)
test_dataset = create_hf_dataset(test_sentences, test_labels)

# Step 5: Create a DatasetDict
datasets = DatasetDict({
    "train": train_dataset,
    "validation": dev_dataset,
    "test": test_dataset
})

# Step 6: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(configuration.tokenizer)

# Step 7: Preprocess the dataset
label_list = list(set([label for sublist in train_labels for label in sublist]))
label_list.sort()

label_to_id = {label: i for i, label in enumerate(label_list)}

tokenized_datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id), batched=True)

# Step 8: Load the model
model = AutoModelForTokenClassification.from_pretrained(configuration.model, num_labels=len(label_list))

# Step 9: Define the metric
metric = evaluate.load("seqeval")

# Step 10: Define training arguments
training_args = TrainingArguments(
    output_dir=f"./results/{configuration.name}",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
    logging_dir="./logs",  # Directory for storing logs
    logging_steps=10,
    save_strategy="epoch",  # Save model at the end of each epoch
)

# Step 11: Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=lambda p: compute_metrics(p, label_list),
)

# Step 12: Train the model
train_result = trainer.train()

# Step 13: Save training results
train_metrics = train_result.metrics
trainer.log_metrics("train", train_metrics)
trainer.save_metrics("train", train_metrics)
trainer.save_state()

# Step 14: Evaluate the model
eval_result = trainer.evaluate()

# Step 15: Save evaluation results
# Access metrics directly from the eval_result dictionary
eval_metrics = eval_result
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# Optional: Print classification report
print(eval_metrics.get('classification_report', 'No classification report available'))

# Step 16: Save the fine-tuned model
model_save_path = f"./models/{configuration.name}/{configuration.model_name}"
os.makedirs(model_save_path, exist_ok=True)
trainer.save_model(model_save_path)

# Optional: Save the tokenizer
tokenizer.save_pretrained(model_save_path)
