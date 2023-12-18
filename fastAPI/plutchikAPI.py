from flask import Flask, request
from flask_restful import Resource, Api
import os
import subprocess
import pandas as pd
import tempfile
from typing import Optional

app = Flask(__name__)
api = Api(app)

class Item:
    def __init__(self, text: str, id_str: Optional[str] = '0', label: Optional[str] = '0'):
        self.text = text
        self.id_str = id_str
        self.label = label

class PredictSentiment(Resource):
    def post(self):
        item = Item(**request.get_json())
        
        # Create a temporary CSV file and write the input text to it
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_csv:
            temp_csv.write('id_str,text,label\n')
            temp_csv.write(f'{item.id_str},{item.text},{item.label}\n')
            temp_csv_path = temp_csv.name
        # Run the command
        cmd = f'python run_classifier.py --load model/transformer_semeval.clf --data {temp_csv_path} --text-key "text" --model "transformer" --write-results result_train_mc.csv --tokenizer-type SentencePieceTokenizer --vocab-size 32000 --tokenizer-path model/ama_32k_tokenizer.model --batch-size 32'
        subprocess.run(cmd, shell=True)

        # Read the results and extract the columns with value 1.0 in row 1
        df = pd.read_csv('result_train_mc.csv')

        # Get the columns names with value 1.0 in row 1
        columns = df.columns[df.iloc[0] == 1.0]
        columns = [column.split(' ')[0] for column in columns]

        os.remove(temp_csv_path)
        return {"emotions": columns}

class PredictSentimentScores(Resource):
    def post(self):
        item = Item(**request.get_json())
        
        # Create a temporary CSV file and write the input text to it
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_csv:
            temp_csv.write('id_str,text,label\n')
            temp_csv.write(f'{item.id_str},{item.text},{item.label}\n')
            temp_csv_path = temp_csv.name
        # Run the command
        cmd = f'python run_classifier.py --load model/transformer_semeval.clf --data {temp_csv_path} --text-key "text" --model "transformer" --write-results result_train_mc.csv --tokenizer-type SentencePieceTokenizer --vocab-size 32000 --tokenizer-path model/ama_32k_tokenizer.model --batch-size 32'
        subprocess.run(cmd, shell=True)

        df = pd.read_csv('result_train_mc.csv')

        # Make a dictionary with prob values for each emotion
        emotions = {}
        for column in df.columns:
            # if column contains word 'prob'
            if 'prob' in column:
                # get the value of the column
                value = df[column].values[0]
                # get the emotion name
                emotion = column.split(' ')[0]
                # add the emotion and value to the emotions dictionary
                emotions[emotion] = value

        os.remove(temp_csv_path)
        return {"emotions": emotions}

api.add_resource(PredictSentimentScores, '/predict_sentiment_scores')
api.add_resource(PredictSentiment, '/predict_sentiment')
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)