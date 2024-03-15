from transformers import pipeline
classifier = pipeline(task="text-classification", model="vibhorag101/roberta-base-emotion-prediction-phr", top_k=None,device_map="auto")
# Can classify 28 emotions
def predict_emotion(text):

    emotions = classifier(text)[0]
    print(emotions)
    res = []
    for emotion in emotions:
        if(emotion["score"] > 0.5):
            res.append(emotion["label"])
    if(len(res) == 0):
        res.append("neutral")
    return(res)

if __name__ == "__main__":
    print(predict_emotion("My son is so caring."))