import requests

# Define the URL of your FastAPI application
url = "http://192.168.3.74:6006/chat_response"

SYSTEM_PROMPT = """
You are a helpful and joyous mental therapy assistant. Always answer as helpfully and cheerfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""
messages = [{"role": "system", "content": SYSTEM_PROMPT}]

data = {
    "messages": messages,
}

def sendRequest(prompt):
    data["messages"].append({"role": "user", "content": prompt.strip()})
    response = requests.post(url, json=data)
    if response.status_code == 200:
        result = response.json()
        return(result)
    else:
        return(f"Request failed with status code: {response.status_code}")

if __name__=="__main__":
    response = sendRequest("Are we ready to chat?")
    messages.append({"role": "assistant", "content": response})
    print(response)
