
# Text2Image & Image2Image Generation Server

This project implements a complete server-client pipeline for **Text-to-Image** and **Image-to-Image** generation using Stable Diffusion. It includes:

- A **gRPC-based backend server** for generating images
- A **Streamlit frontend** for interaction
- Full support for both `text2image` and `img2img` generation modes
- Resource usage tracking and visualization (CPU/GPU/Memory)

---

## 🧠 Features

- 🔠 **Text2Image**: Enter a text prompt to generate a new image
- 🖼️ **Image2Image**: Provide a reference image and modify it with a prompt
- ⚡ gRPC backend optimized for performance
- 📊 Resource usage tracking and live plots
- 🖥️ Web interface using **Streamlit**

---

## 🔧 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Ziptoze/NLP-Project
cd NLP-Project
```

### 2. Create and Activate a Virtual Environment

If no virtual environment is present, create one:

```bash
python -m venv venv
```

Activate it:

```bash
# Windows
.env\Scriptsctivate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Compile gRPC Protobuf File

```bash
python -m grpc_tools.protoc -I app --python_out=app --grpc_python_out=app app/text2image.proto
```

---

## 🚀 Running the Project

### 🧠 Backend (gRPC Server)

```bash
python app/server.py
```

This launches the server and waits for incoming image generation requests.

### 🖼️ Frontend (Streamlit App)

```bash
streamlit run app/streamlit_app.py
```

This launches a web UI to send requests and view results.

---

## 🖋️ Functionalities

### 🔡 Text2Image

- Prompt-based generation
- Optional parameters like `guidance_scale`, `steps`, `seed`, etc.
- Implemented in `server.py` and triggered from `streamlit_app.py`

### 🖼️ Image2Image

- Upload a reference image
- Modify it using a prompt
- Adjust strength, noise, and other parameters
- Uses same gRPC protocol as `text2image`

---

## 📈 Resource Usage and Performance Tracking

The backend measures and logs:

- CPU Usage
- GPU Usage (if available via `torch.cuda`)
- RAM usage (via `psutil`)
- Time taken for generation

You can optionally visualize these using `matplotlib` or Streamlit built-in charts.

> Tip: Add plots in `streamlit_app.py` to display performance over time.

---

## 🧪 Example Commands

```bash
# Activate environment
.env\Scriptsctivate

# Install dependencies
pip install -r requirements.txt

# Compile protobuf
python -m grpc_tools.protoc -I app --python_out=app --grpc_python_out=app app/text2image.proto

# Run server
python app/server.py

# Run frontend
streamlit run app/streamlit_app.py
```

---

## 📂 Project Structure

```
text2img-grpc-server/
│
├── app/
│   ├── text2image.proto          # gRPC definition
│   ├── server.py                 # Backend server logic
│   ├── streamlit_app.py          # Streamlit UI
│   ├── model_loader.py           # Model loading code
│   ├── utils.py                  # Utility functions
│   └── ...
│
├── requirements.txt
├── README.md
└── venv/ (created after setup)
```

---

## 🛠️ Requirements

- Python 3.9+
- Streamlit
- torch
- torchvision
- diffusers
- transformers
- grpcio
- grpcio-tools
- Pillow
- psutil
- matplotlib (optional for plots)

> Full list in `requirements.txt`

---

## 📌 Notes

- Make sure to have internet access on the first run to download model weights
- You can adjust model checkpoint in `model_loader.py`
- Works on both CPU and GPU (GPU highly recommended for performance)

---

## 📬 Contact

For bugs, issues, or suggestions, open an issue or email `your.email@example.com`.

---

## 🖼️ Preview

![Example UI Screenshot](https://yourimagehost.com/sample_ui.png)  
*Text prompt: "A futuristic city skyline at sunset"*

---

## 📜 License

MIT License. See `LICENSE` for details.
