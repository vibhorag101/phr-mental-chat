# phr-mental-chat
A mental therapy chatbot

### Model and dataset details:
- The original dataset is accessible at [here](https://huggingface.co/datasets/jerryjalapeno/nart-100k-synthetic)
- The modified dataset for llama-2 format is accessible at [here](https://huggingface.co/datasets/vibhorag101/phr-mental-therapy-dataset-conversational-format)
- The finetuned llama-2-7b-chat-hf model is accessible at [here](https://huggingface.co/vibhorag101/llama-2-7b-chat-hf-phr_mental_therapy_v2)

### To build the docker image, run following commands:
```
docker build -t {image_name} .
docker run --gpus all -p 5555:5555  -itd --name {container_name} {image_name}
```
- Chatbot will be accessible at `http://{server_ip}:5555/gradio`
- Chatbot API will be accessible at `http://{server_ip}:5555/generate`
- Roberta Emotion Classification api will be accessible at `http://{server_ip}:5555/emotion`
- Suicide Prediction api will be accessible at `http://{server_ip}:5555/suicide`
- Plutchik Emotion API available at `http://{server_ip}:5000/predict_sentiment` and `http://{server_ip}:5000/predict_sentiment_scores`
- The API docs are available at `http://{server_ip}:5555/docs`
