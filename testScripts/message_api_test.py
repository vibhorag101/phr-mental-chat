from openai import OpenAI

# init the client but point it to TGI
client = OpenAI(
    base_url="http://192.168.3.74:8080/v1",
    api_key="-"
)

SYSTEM_PROMPT = """
You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
history = [{"role": "system", "content": SYSTEM_PROMPT}]
def chat_with_tgi(prompt, history=history):
    history.append({"role": "user", "content": prompt.strip()})
    chat_completion = client.chat.completions.create(
        model="tgi",
        # add prompt to the messages
        messages=history,
    )
    response = chat_completion.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": response})
    return response

while True:
    input_text = input("You: ")
    if input_text == "exit":
        break
    print("TGI: ", chat_with_tgi(input_text, history=history))


# apply the jinja2 template to the messages
# from jinja2 import Template
# template = Template("{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}")

# # apply the jinja2 template to the messages
# print(template.render(messages=messages, bos_token="<s>", eos_token="</s>"))