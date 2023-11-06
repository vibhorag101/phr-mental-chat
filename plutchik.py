import requests
import json
import pandas as pd

url = 'http://192.168.3.74:5000/predict_sentiment'

# Define the text you want to analyze
text_to_analyze = 'I am very happy today! The atmosphere looks cheerful '

# Create a dictionary with the text
data = {'text': text_to_analyze}

# Send a POST request to the API with the data as JSON
response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

# Check if the request was successful
if response.status_code == 200:
    result = response.json()
    print(result)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame([result])

    # Save the DataFrame to a CSV file
    df.to_csv('response.csv', index=False)
else:
    print(f"Request failed with status code: {response.status_code}")
    print(response.text)  # Print the response content for debugging
