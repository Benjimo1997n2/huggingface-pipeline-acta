# Step 1: Install the necessary libraries (if not already installed)
# pip install transformers datasets evaluate

# Step 2: Import the necessary libraries
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
import evaluate

# Import custom functions from util.py
from util import read_conll_file, create_hf_dataset, tokenize_and_align_labels, compute_metrics

# Step 3: Read the CoNLL files
train_sentences, train_labels = read_conll_file("datasets/abstrct/train.conll")
dev_sentences, dev_labels = read_conll_file("datasets/abstrct/dev.conll")
test_sentences, test_labels = read_conll_file("datasets/abstrct/test.conll")

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
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")

# Step 7: Preprocess the dataset
label_list = list(set([label for sublist in train_labels for label in sublist]))
label_list.sort()

label_to_id = {label: i for i, label in enumerate(label_list)}

tokenized_datasets = datasets.map(lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id), batched=True)

# Step 8: Load the model
model = AutoModelForTokenClassification.from_pretrained("allenai/scibert_scivocab_uncased", num_labels=len(label_list))

# Step 9: Define the metric
metric = evaluate.load("seqeval")

# Step 10: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
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
trainer.train()

# Step 13: Evaluate the model
trainer.evaluate()