#!/bin/bash
# Launch the SEC-RAG Gradio app.
# Optional: SEC_RAG_PORT=7860 (default), SEC_RAG_SHARE=1 for a temporary public link.
cd "$(dirname "$0")" || exit 1
exec python app.py
