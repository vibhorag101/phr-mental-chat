import os
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments, logging, pipeline,EarlyStoppingCallback,IntervalStrategy)
from trl import SFTTrainer

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

base_model_name = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "vibhorag101/phr-mental-therapy-dataset-conversational-format-1024-tokens"
new_model_name = "llama-2-7b-chat-hf-phr_mental_therapy-3"
output_dir = "./results"

## Memory Saving Hyperparameters
num_train_epochs = 1
per_device_train_batch_size = 1
per_device_eval_batch_size = 8
max_seq_length = 1024
## It says the effective batch size = per_device_train_batch_size * gradient_accumulation_steps, so we can increase the effective ##batch size without running out of memory
gradient_accumulation_steps=1
## It saves memory by checkpointing the gradients (set to True if memory is an issue)
gradient_checkpointing = False

dataset = load_dataset(dataset_name)

# total_dataset_size = 20000
# train_size = int(0.7 * total_dataset_size)
# val_size = int(0.15 * total_dataset_size)
# dataset["train"] = dataset["train"].select(range(train_size))
# dataset["val"] = dataset["val"].select(range(val_size))

# QLoRA parameters and bits and bytes
# Hyperparameters set as recommended in the qLoRA paper
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False
fp16 = False
bf16 = True

### Step Parameters ###
save_steps = 1000
logging_steps = 25
eval_steps = 1000

### Model Parameters ###
max_grad_norm = 0.3
learning_rate = 2e-5
weight_decay = 0.001
optim = "paged_adamw_32bit" ## paged optim to save memory 32bit or 8bit
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
device_map = {"": 0}
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    # hub_model_id=new_model_name, ## the name of repo in huggingface pushed by Trainer otherwise output_dir is used
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy="steps",
    save_steps=save_steps,
    eval_steps=eval_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    resume_from_checkpoint=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    optim=optim,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
tokenizer.add_special_tokens({"pad_token": "<pad>"})
model.resize_token_embeddings(len(tokenizer))

#Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    peft_config=peft_config,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)]
)
trainer.train()
trainer.save_model(new_model_name)
trainer.push_to_hub(new_model_name)