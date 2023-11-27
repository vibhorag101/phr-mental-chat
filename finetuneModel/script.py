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
finetune_model_name = "roberta-base-emotion-prediction-phr-3"

# %%
label2id = {
    "anger":0,
    "anticipation":1,
    "disgust":1,
    "fear":1,
    "joy":1,
    "love":1,
    "optimism":1,
    "pessimism":1,
    "sadness":1,
    "surprise":1,
    "trust":1
}

id2label = {
    0:"anger",
    1:"anticipation",
    2:"disgust",
    3:"fear",
    4:"joy",
    5:"love",
    6:"optimism",
    7:"pessimism",
    8:"sadness",
    9:"surprise",
    10:"trust"
}

# %%
tokeniser = AutoTokenizer.from_pretrained("roberta-base",problem_type ="multi_label_classification")
model = AutoModelForSequenceClassification.from_pretrained("google/electra-base-discriminator", problem_type ="multi_label_classification", num_labels=11,label2id=label2id,id2label=id2label)

# %%
dataset = load_dataset("vibhorag101/sem_eval_2018_task_1_english_cleaned_labels")

# %%
def tokeniseDataset(dataset):
    return(tokeniser(dataset["text"],padding="max_length",truncation=True))

#
dataset.set_format("torch")
dataset = dataset.map(lambda x : {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"]).rename_column("float_labels", "labels")

column_names = dataset["train"].column_names
column_names.remove("labels")
tokenisedDataset = dataset.map(tokeniseDataset,batched=True,remove_columns=column_names)

# %%
trainTokeniseDataset = tokenisedDataset["train"]
testTokenisedDataset = tokenisedDataset["test"]
valTokenisedDataset = tokenisedDataset["test"]

# %%
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score,precision_score,recall_score

def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    sigmoid_pred = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into predicted labels
    y_pred  = np.where(sigmoid_pred > threshold, 1, 0)

    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)

    metrics = { 'accuracy': accuracy,
                'micro_precision': precision,
                'micro_recall': recall,
                'micro_f1': f1_micro_average,
                'micro_roc_auc': roc_auc,
                }
    return metrics

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return multi_label_metrics(logits, labels)


# %%
wandb.login()
wandb.init(project="huggingface", entity="vibhor20349", name=finetune_model_name)

# %%
training_args = TrainingArguments(
    output_dir=finetune_model_name,
    report_to = 'wandb',
    learning_rate=2e-5, # recommended in roberta paper
    num_train_epochs=3, #recommended in bert paper
    per_device_train_batch_size=16, # recommended in roberta paper
    per_device_eval_batch_size=16, # recommended in roberta paper
    evaluation_strategy="steps",
    eval_steps = 100,
    save_strategy="steps",
    save_steps=100,
    load_best_model_at_end=True,
    logging_steps=100,
    metric_for_best_model="eval_micro_f1",
    push_to_hub=True #to clone the training repo just before starting to train, so push to hub works
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
# print(trainer.predict(testTokenisedDataset))
trainer.evaluate(eval_dataset=testTokenisedDataset)

# %%
trainer.train()
# #trainer.train(resume_from_checkpoint=True)

# %%
trainer.save_model(finetune_model_name)

# %%
trainer.push_to_hub(finetune_model_name)

# %%
### IF we want to make predictions
# input_text = "You are an extremely good person. Thank you so much"
# inputs = tokeniser(input_text, return_tensors="pt")

# # Make predictions
# with torch.no_grad():
#     ## BERT expects the keyword agruments "token_type_ids" and "attention_mask" in input.
#     # so we convert inputs dictionary to keyword arguments using ** before passing to the model
#     inputs.to('cuda')
#     outputs = model(**inputs)
#     ## output of Bert contains logits for each class, pooled output and hidden states and attentions

# # Extract the predicted logits(raw values for each class)
# logits = outputs.logits
# print(outputs)

# ## logits are just raw values for each class. To get probabilities we use softmax
# sigmoid_logits = torch.nn.functional.sigmoid(logits)
# predictions = np.where(sigmoid_logits.cpu() > 0.5, 1, 0)

# # Get the predicted labels from id2label dictionary
# label_predictions = []
# for pred in predictions:
#     label_predictions.append([id2label[i] for i, val in enumerate(pred) if val])
# print(label_predictions)



