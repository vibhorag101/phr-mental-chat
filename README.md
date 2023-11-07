# phr-mental-chat
A mental therapy chatbot

### To build the docker image run following commands:
```
docker build -t {image_name} .
docker run --gpus all -p 5555:5555 {image_name}
```
- Chatbot will be accessible at `http://{server_ip:5555/gradio}`
- Chatbot API will be accessible at `http://{server_ip:5555/generate}`
