import re
from threading import Thread
from typing import Iterator, List,Dict

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer, pipeline,BitsAndBytesConfig)

model_name = "vibhorag101/llama-2-7b-chat-hf-phr_mental_therapy_v2"
use_4bit=True
device_map = {"": 0}
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_input_token_length(messages) -> int:
    return(len(tokenizer.apply_chat_template(messages)))

def get_LLAMA_response_stream(
        messages:List[Dict[str, str]],
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> Iterator[str]:
    
    prompt = tokenizer.apply_chat_template(messages,tokenize=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to('cuda')
    streamer = TextIteratorStreamer(tokenizer,
                                    timeout=10.,
                                    skip_prompt=True,
                                    skip_special_tokens=True)
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)

def get_LLAMA_response(
        messages,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> str:
    
    prompt = tokenizer.apply_chat_template(messages,tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs["input_ids"]
    output_ids = model.generate(
    **inputs,
    max_length=1024,
    do_sample=True,
    top_p=0.95,
    top_k=50,
    temperature=1)
    output_text = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
    return output_text