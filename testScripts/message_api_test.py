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
        messages=history,
        max_tokens=1024,
        temperature=1,
        top_p=0.9
    )
    response = chat_completion.choices[0].message.content.strip()
    history.append({"role": "assistant", "content": response})
    return response

while True:
    input_text = input("You: ")
    if input_text == "exit":
        break
    print("TGI: ", chat_with_tgi(input_text, history=history))