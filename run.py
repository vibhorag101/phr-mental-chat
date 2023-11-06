from api import app
from app import demo
import gradio as gr
import uvicorn

if __name__ == "__main__":
    # run the gradion app on the fastapi server. access using/gradio endpoint 
    app = gr.mount_gradio_app(app,demo.queue(max_size=20),path="/gradio")
    uvicorn.run("run:app", host="0.0.0.0", port=5555)