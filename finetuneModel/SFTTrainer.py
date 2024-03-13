# %%
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments, logging, pipeline)
from trl import SFTTrainer

base_model = "meta-llama/Llama-2-7b-chat-hf"
dataset_name = "vibhorag101/phr-mental-therapy-dataset-conversational-format-mini"
new_model = "llama-2-7b-chat-hf-phr_mental_therapy-2"

# Hyperparameters
num_train_epochs = 3
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
max_seq_length = 1024

# %%
dataset = load_dataset(dataset_name)

# QLoRA parameters and bits and bytes
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
save_steps = 100
logging_steps = 25
eval_steps = 100

### Model Parameters ###
output_dir = "./results"
max_grad_norm = 0.3
learning_rate = 2e-5
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
device_map = {"": 0}
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

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
    optim=optim,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map=device_map,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(base_model)
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
)
trainer.train()
trainer.save_model(new_model)
trainer.push_to_hub(new_model)