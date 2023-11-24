# %%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, load_metric,load_dataset
import evaluate

# %% [markdown]
# - We must tokenise the input before feeding to the model. We use the tokenizer from the pretrained model.
# - The dataset must have the label column named as 'label', to be trained using Trainer API. It is preferable to keep the text column as 'text'. However since we are tokenising the text column, it can be named anything, as we just pass the tokenised columns input_ids and attention_mask to the model.

# %%
# for huggingface pipeline to give text labels instead of numbers
id2label = {0: "suicide", 1:"non-suicide"}
label2id = {"suicide":0, "non-suicide":1}

# %%
tokeniser = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2,label2id=label2id,id2label=id2label)

# %%
dataset = load_dataset("vibhorag101/phr_suicide_prediction_dataset_clean")

# %%
def tokeniseDataset(dataset):
    return(tokeniser(dataset["text"],padding="max_length",truncation=True))

def convertLabel2ID(dataset):
    dataset['label'] = label2id[dataset['label']]
    return dataset
    
dataset = dataset.map(convertLabel2ID) 
tokenisedDataset = dataset.map(tokeniseDataset,batched=True)

trainTokeniseDataset = tokenisedDataset["train"]
tempTokenisedDataset= tokenisedDataset["test"]
tempTokenisedDataset = tempTokenisedDataset.train_test_split(test_size=0.5)
testTokenisedDataset = tempTokenisedDataset["train"]
valTokenisedDataset = tempTokenisedDataset["test"]
# print(trainTokeniseDataset)
# print(valTokenisedDataset)
# print(testTokenisedDataset)

# %%
def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_rec = evaluate.load("recall")
    metric_pre = evaluate.load("precision")
    metric_f1 = evaluate.load("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_acc.compute(predictions=predictions, references=labels)
    recall = metric_rec.compute(predictions=predictions, references=labels)
    precision = metric_pre.compute(predictions=predictions, references=labels)
    f1 = metric_f1.compute(predictions=predictions, references=labels)

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}


# %%
wandb.login()
wandb.init(project="huggingface", entity="vibhor20349", name="roberta-suicide-prediction-cleaned")

# %%
finetune_model_name = "PHR_Suicide_Prediction_Roberta_Cleaned"
training_args = TrainingArguments(
    output_dir=finetune_model_name,
    report_to = 'wandb',
    learning_rate=2e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps = 1000,
    save_strategy="steps",
    save_steps=1000,
    load_best_model_at_end=True,
    push_to_hub=True
)

# %%
trainer = Trainer(
    model= model,
    args=training_args,
    train_dataset=trainTokeniseDataset,
    eval_dataset=valTokenisedDataset,
    compute_metrics=compute_metrics,
    tokenizer=tokeniser
)

# %%
trainer.evaluate(eval_dataset=testTokenisedDataset)

# %%
trainer.train()

# %%
trainer.save_model(finetune_model_name)

# %%
trainer.push_to_hub(finetune_model_name)

# %%
# if above is not working use the moodel push
# model.push_to_hub("PHR_Suicide_Prediction_Roberta")
# tokeniser.push_to_hub("PHR_Suicide_Prediction_Roberta")


