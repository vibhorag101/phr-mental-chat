import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, load_metric,load_dataset
import evaluate

id2label = {0: "suicide", 1:"non-suicide"}
label2id = {"suicide":0, "non-suicide":1}

finetune_model_name = "PHR_Suicide_Prediction_Roberta_Cleaned_Light"

tokeniser = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2,label2id=label2id,id2label=id2label)

dataset = load_dataset("vibhorag101/phr_suicide_prediction_dataset_clean_light")

def tokeniseDataset(dataset):
    return(tokeniser(dataset["text"],padding="max_length",truncation=True))

def convertLabel2ID(dataset):
    dataset['label'] = label2id[dataset['label']]
    return dataset
    
dataset = dataset.map(convertLabel2ID) 
tokenisedDataset = dataset.map(tokeniseDataset,batched=True)

trainTokeniseDataset = tokenisedDataset["train"]
valTOkeniseDataset= tokenisedDataset["val"]

def compute_metrics(eval_pred):
    metric_acc = evaluate.load("accuracy")
    metric_rec = evaluate.load("recall")
    metric_pre = evaluate.load("precision")
    metric_f1 = evaluate.load("f1")

    # here the model just give logits, labels are passed from test/val dataset
    logits, labels = eval_pred
    
    # for binary classification we can just take argmax of logits, but for multi-class classification we need to use softmax
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = metric_acc.compute(predictions=predictions, references=labels)['accuracy']
    recall = metric_rec.compute(predictions=predictions, references=labels)['recall']
    precision = metric_pre.compute(predictions=predictions, references=labels)['precision']
    f1 = metric_f1.compute(predictions=predictions, references=labels)['f1']

    return {"accuracy": accuracy, "recall": recall, "precision": precision, "f1": f1}


wandb.login()
wandb.init(project="huggingface", entity="vibhor20349", name=finetune_model_name)

training_args = TrainingArguments(
    per_device_train_batch_size=16, # recommended in roberta paper
    per_device_eval_batch_size=32, # recommended in roberta paper
    learning_rate=2e-5, # recommended in roberta paper
    num_train_epochs=3, #recommended in bert paper
    eval_steps =500,
    save_steps=500,
    logging_steps=25,    
    warmup_ratio=0.06, # recommended in roberta paper
    weight_decay=0.1, # recommended in roberta paper
    load_best_model_at_end=True,
    resume_from_checkpoint=True,
    metric_for_best_model="eval_f1",
    evaluation_strategy="steps",
    save_strategy="steps",
    output_dir=finetune_model_name,
    report_to = 'wandb'
    # push_to_hub=True, To push every checkpoint to the hub
)

## Here tokeniser is passed to the trainer, but it is for only pushing to hub and not for tokenising the dataset
trainer = Trainer(
    model= model,
    args=training_args,
    train_dataset=trainTokeniseDataset,
    eval_dataset=valTOkeniseDataset,
    compute_metrics=compute_metrics,
    tokenizer=tokeniser,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5,early_stopping_threshold=0.001)]
)

# trainer.evaluate(eval_dataset=valTOkeniseDataset)

trainer.train()
trainer.save_model(finetune_model_name)
trainer.push_to_hub(finetune_model_name)



