"""Entry point: python -m audiobook"""

import gradio as gr

from core import get_device
from .ui import build_app

if __name__ == "__main__":
    print(f"Device: {get_device()}")
    print("Models load on first generation.")
    app = build_app()
    app.launch(theme=gr.themes.Soft())
