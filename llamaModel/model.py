import re
from threading import Thread
from typing import Iterator
from suicideModel.model import predict_suicide

import torch
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          TextIteratorStreamer, pipeline)

model_id = 'vibhorag101/llama-2-13b-chat-hf-phr_mental_therapy'

if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float16,
        load_in_4bit=True,
        device_map='auto'
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,use_fast=False,add_eos_token=True)
tokenizer.pad_token_id = 18610


def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)


def get_input_token_length(message: str, chat_history: list[tuple[str, str]], system_prompt: str) -> int:
    prompt = get_prompt(message, chat_history, system_prompt)
    input_ids = tokenizer([prompt], return_tensors='np', add_special_tokens=False)['input_ids']
    return input_ids.shape[-1]


def run(message: str,
        chat_history: list[tuple[str, str]],
        system_prompt: str,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = 50) -> Iterator[str]:
    
    # check if message contains any variations of the word suicide. If they do, then simply tell to consult a doctor.
    # as the things can be serious.
    if predict_suicide(message)=='suicide':
        yield "I am sorry that you are feeling this way. You need a specialist help. Please consult a nearby doctor."
    else:
        prompt = get_prompt(message, chat_history, system_prompt)
        
        inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

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
