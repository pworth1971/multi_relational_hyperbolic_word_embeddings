import torch

from transformers import (
    BertForSequenceClassification, BertTokenizer,
    RobertaForSequenceClassification, RobertaTokenizer,
    XLNetForSequenceClassification, XLNetTokenizer,
    ElectraForSequenceClassification, ElectraTokenizer,
    Trainer, TrainingArguments
)

from datasets import load_dataset

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import pandas as pd

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Display the results as a table
import matplotlib.pyplot as plt
import seaborn as sns

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load the datasets
datasets = {
    "sentiment140": load_dataset("sentiment140"),
    "amazon_polarity": load_dataset("amazon_polarity"),
    "imdb": load_dataset("imdb")
}

# Define the models and tokenizers
models_and_tokenizers = {
    "xlnet": (XLNetForSequenceClassification.from_pretrained("xlnet-base-cased"), XLNetTokenizer.from_pretrained("xlnet-base-cased")),
    "electra": (ElectraForSequenceClassification.from_pretrained("google/electra-small-discriminator"), ElectraTokenizer.from_pretrained("google/electra-small-discriminator")),
    "bert": (BertForSequenceClassification.from_pretrained("bert-base-uncased"), BertTokenizer.from_pretrained("bert-base-uncased")),
    "roberta": (RobertaForSequenceClassification.from_pretrained("roberta-base"), RobertaTokenizer.from_pretrained("roberta-base")),
}

def preprocess_function(examples, tokenizer, max_length=128):
    texts = examples['text']
    cleaned_texts = []
    for text in texts:
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stop_words]
        cleaned_text = ' '.join(filtered_words)
        cleaned_texts.append(cleaned_text)
    return tokenizer(cleaned_texts, padding="max_length", truncation=True, max_length=max_length)

def add_labels(examples, label_column):
    return {'labels': examples[label_column]}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Store results
results = []

# Training and evaluation loop for each dataset and model
for dataset_name, dataset in datasets.items():

    print()
    print("------------------------------------------------------------------------------------------------------------------------")
    print(f"*** Training on dataset: {dataset_name} ***")
    print("------------------------------------------------------")

    dataset = dataset['train'].train_test_split(test_size=0.2)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    
    # Check dataset columns
    print(f"Dataset columns: {train_dataset.column_names}")

    # Identify the label column (defaulting to 'label')
    label_column = 'label' if 'label' in train_dataset.column_names else 'sentiment'
    print(f"Label column: {label_column}")

    for model_name, (model, tokenizer) in models_and_tokenizers.items():

        print()
        print(f"-- USING MODEL {model_name} --")

        tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        tokenized_eval = eval_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
        
        # Ensure labels are included
        tokenized_train = tokenized_train.map(lambda examples: add_labels(examples, label_column), batched=True)
        tokenized_eval = tokenized_eval.map(lambda examples: add_labels(examples, label_column), batched=True)
        
        # Set format for PyTorch
        tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        tokenized_eval.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=lambda p: {
                "accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1)),
                "precision": precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average='binary')[0],
                "recall": precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average='binary')[1],
                "f1": precision_recall_fscore_support(p.label_ids, p.predictions.argmax(-1), average='binary')[2],
            }
        )

        # Train the model
        trainer.train()

        # Evaluate the model
        eval_result = trainer.evaluate()

        # Store the result
        results.append({
            "dataset": dataset_name,
            "model": model_name,
            "accuracy": eval_result['eval_accuracy'],
            "precision": eval_result['eval_precision'],
            "recall": eval_result['eval_recall'],
            "f1": eval_result['eval_f1']
        })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)

# Save results to a CSV file
results_df.to_csv("output/bc_results_summary.csv", index=False)

plt.figure(figsize=(20, 12))
sns.set(style="whitegrid")

results_melted = results_df.melt(id_vars=["dataset", "model"], var_name="metric", value_name="value")
g = sns.catplot(x="model", y="value", hue="dataset", col="metric", data=results_melted, kind="bar", height=4, aspect=1.5)

# Save the plot to a file
g.savefig("output/bc_results_summary_plot.png")

print("Training and evaluation complete.")
print("----------------------------------------------------------------------------------------")
