import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments, logging, pipeline)
from trl import SFTTrainer

base_model_name = "meta-llama/Llama-2-7b-chat-hf"
new_model_name = "llama-2-7b-chat-hf-phr_mental_therapy-3"
use_4bit=True
device_map = {"": 0}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

## QLoRA Inference with adapter

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
)
## To account for pad tokens added to the model while fine-tuning
tokenizer = AutoTokenizer.from_pretrained(new_model_name)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, new_model_name)


conv = [ { "content": "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.", "role": "system" }]

def generate_response(conv,model,tokenizer):
    prompt = tokenizer.apply_chat_template(conv,tokenize=False)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    input_ids = input_ids.to('cuda')
    output_ids = model.generate(
    input_ids,
    max_length=1024,
    do_sample=True,
    top_p=0.95,
    top_k=60,
    temperature=0.9)
    output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text

def chat_with_model(user_input,conv,model,tokenizer):
    conv.append({"content": user_input, "role": "user"})
    response = generate_response(conv,model,tokenizer)
    print("Model:", response)
    conv.append({"content": response, "role": "assistant"})


if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        chat_with_model(user_input,conv,model,tokenizer)


