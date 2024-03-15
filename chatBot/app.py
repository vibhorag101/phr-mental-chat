import os
from threading import Thread
from typing import Iterator
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI
import requests, json

MAX_MAX_NEW_TOKENS = 2048
DEFAULT_MAX_NEW_TOKENS = 1024
MAX_INPUT_TOKEN_LENGTH = int(os.getenv("MAX_INPUT_TOKEN_LENGTH", "4096"))
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

DESCRIPTION = """
# LLama-2-Mental-Therapy-Chatbot
"""

client = OpenAI(
    base_url="http://192.168.3.74:8080/v1",
    api_key="-"
)

def predict_suicide(text):
    url = 'http://192.168.3.74:6006/suicide'
    data = {'message': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        result = response.json()
        return(result)
    
def predict_threat(text):
    url = 'http://192.168.3.74:6006/threat'
    data = {'message': text}
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})
    if response.status_code == 200:
        result = response.json()
        return(result)
    
def response_guard(message):
    if os.getenv("PREDICT_SUICIDE")=="True" and predict_suicide(message)=='suicide':
        return("I am sorry that you are feeling this way. You need a specialist help. Please consult a nearby doctor.")
    if os.getenv("PREDICT_THREAT")=="True" and predict_threat(message)=='threat':
        return("We detected unlawful language and intentions in the conversation.")

    return("safe")
    
def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    system_prompt: str,
    max_new_tokens: int = 1024,
    temperature: float = 1,
    top_p: float = 0.9,
) -> Iterator[str]:
    llmGuardCheck = response_guard(message)
    if(llmGuardCheck != "safe"):
        raise gr.Error(llmGuardCheck)
        yield(llmGuardCheck)
    else:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for user, assistant in chat_history:
            messages.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
        messages.append({"role": "user", "content": message})
        chat_completion = client.chat.completions.create(
            model="tgi", messages=messages, stream=True,max_tokens=max_new_tokens,temperature=temperature,top_p=top_p
        )
        response = ""
        first_chunk = True
        for chunk in chat_completion:
            token = chunk.choices[0].delta.content
            if first_chunk:
                token= token.strip() ## the first token Has a leading space, due to some bug in TGI
                response += token
                yield response
                first_chunk = False
            else:
                if token!="</s>":
                    response += token
                    yield response


chat_interface = gr.ChatInterface(
    fn=generate,
    additional_inputs=[
        gr.Textbox(label='System Prompt',
                                   value=DEFAULT_SYSTEM_PROMPT,
                                   lines=6),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=1,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.95,
        ),
    ],
    stop_btn="Stop",
)

with gr.Blocks(css="style.css") as demo:
    gr.Markdown(DESCRIPTION)
    chat_interface.render()
    
if __name__ == "__main__":
    demo.queue(max_size=20).launch()