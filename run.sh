#!/bin/bash

source venv/bin/activate

# Run server
gnome-terminal -- bash -c "python app/server.py; exec bash"

# Run frontend
gnome-terminal -- bash -c "streamlit run frontend/streamlit_app.py; exec bash"




Set-ExecutionPolicy Unrestricted -Scope Process
get-ExecutionPolicy


pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

cd /
E:
cd E:\Python\text2IMAGE_microservice

.\venv\Scripts\Activate
pip install -r requirements.txt
python -m grpc_tools.protoc -I app --python_out=app --grpc_python_out=app app/text2image.proto
python app/server.py



streamlit run app/streamlit_app.py
