from fastAPI.api import app
import gradio as gr
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    if(os.getenv("USE_LOCAL_MODEL")=="True"):
        from chatBot.app_local import demo
    else:
        from chatBot.app import demo
    # run the gradion app on the fastapi server. access using/gradio endpoint
    app = gr.mount_gradio_app(app,demo.queue(max_size=20),path="/gradio")
    uvicorn.run("run:app", host=os.getenv("HOST"), port=int(os.getenv("PORT")))