#!/bin/bash
# running inference using https://github.com/huggingface/text-generation-inference
# make executable using chmod u+x run_tgi.sh
# run using ./run_tgi.sh
# pull using docker pull ghcr.io/huggingface/text-generation-inference:sha-8a5bcba
# add the image_id to image_name variable
token="enter-token-here"
volume=$PWD/data
container_name=phr_ai_api
image_name=287785ff9b29
model=vibhorag101/llama-2-7b-chat-hf-phr_mental_therapy_v2
MAX_TOTAL_TOKENS=4096
MAX_INPUT_LENGTH=2048

# Running the Docker command
docker run --gpus all --shm-size 1g -e HUGGING_FACE_HUB_TOKEN=$token -p 8080:80  -v $volume:/data --name $container_name -itd $image_name --model-id $model --quantize bitsandbytes-nf4 --max-total-tokens $MAX_TOTAL_TOKENS --max-input-length $MAX_INPUT_LENGTH