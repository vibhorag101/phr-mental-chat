import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, HfArgumentParser,
                          TrainingArguments, logging, pipeline)
from trl import SFTTrainer

model_name = "vibhorag101/llama-2-7b-chat-hf-phr_mental_therapy_v2"
use_4bit=True
device_map = {"": 0}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

## QLoRA Inference with adapter

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


conv = [ { "content": "You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.", "role": "system" }]

def generate_response(conv,model,tokenizer):
    # if we set tokenize=False, only the chat template is applied.
    # no tokenization is done, and we get string insted of token ids.
    prompt = tokenizer.apply_chat_template(conv,tokenize=False)
    # THe model.generate() takes token input_ids, attention_mask, and other parameters
    # and returns output token ids of the generated response.
    # although we can also pass only input_ids using tokenizer.encode(prompt, return_tensors="pt")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=1024,
    do_sample=True,
    top_p=0.95,
    top_k=30,
    temperature=1)
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


