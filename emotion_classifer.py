from transformers import pipeline

# Using https://huggingface.co/SamLowe/roberta-base-go_emotions
# Can classify 28 emotions
def predict_emotion(text):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None,device_map="auto")
    emotions = classifier(text)
    return(emotions[0][0]["label"])

if __name__ == "__main__":
    print(predict_emotion("My son is so cute!"))