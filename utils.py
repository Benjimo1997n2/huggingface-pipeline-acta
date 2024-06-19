from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def read_conll_file(file_path):
    sentences = []
    labels = []
    sentence = []
    label = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence = []
                    label = []
            else:
                parts = line.strip().split()
                sentence.append(parts[0])
                label.append(parts[-1])

    # Append the last sentence if the file doesn't end with a newline
    if sentence:
        sentences.append(sentence)
        labels.append(label)

    return sentences, labels

def create_hf_dataset(sentences, labels):
    data = {"tokens": sentences, "ner_tags": labels}
    return Dataset.from_dict(data)

def tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length=256):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, padding='max_length', max_length=max_length, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten the lists
    true_labels_flat = [item for sublist in true_labels for item in sublist]
    true_predictions_flat = [item for sublist in true_predictions for item in sublist]

    # Compute the metrics
    accuracy = accuracy_score(true_labels_flat, true_predictions_flat)
    f1 = f1_score(true_labels_flat, true_predictions_flat, average="weighted")
    macro_f1 = f1_score(true_labels_flat, true_predictions_flat, average="macro")

    classification_rep = classification_report(true_labels_flat, true_predictions_flat, target_names=label_list)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "macro_f1": macro_f1,
        "classification_report": classification_rep,
    }