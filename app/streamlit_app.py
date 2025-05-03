# streamlit_app.py

import streamlit as st
import grpc
import text2image_pb2 as pb2
import text2image_pb2_grpc as pb2_grpc
import enhancement
import sys
from speech import recognize_voice


if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

# Load models once
@st.cache_resource
def load_models():
    enhancement.download_models()

load_models()

def get_stub():
    channel = grpc.insecure_channel('localhost:50051')
    return pb2_grpc.Text2ImageServiceStub(channel)

st.title("🎨 Text to Image Generator")

# --- Language Dropdown ---
lang = st.selectbox("Prompt Language", options=["English", "Urdu"])

# --- Prompt Input ---
prompt_input = st.text_input("Enter your main prompt:", "")
context = st.text_area("Enter additional context (optional):", "")
enhance_prompt = st.checkbox("✨ Enhance Prompt")

from speech import recognize_voice

if st.button("🎙 Record Prompt (Voice)"):
    with st.spinner("Listening..."):
        text = recognize_voice(duration=5)
        st.text_area("Recognized Prompt", text, key="voice_prompt")
        st.session_state["final_prompt"] = text

# --- Placeholder for outputs ---
translated_prompt = ""
final_prompt = prompt_input

if st.button("Translate / Enhance Prompt"):
    if prompt_input:
        if lang == "Urdu":
            translated_prompt = enhancement.translate_to_english(prompt_input)
            st.markdown(f"**Translated to English:** `{translated_prompt}`")
            final_prompt = translated_prompt
        else:
            final_prompt = prompt_input

        if enhance_prompt:
            final_prompt = enhancement.enhance_prompt(final_prompt)
            st.markdown(f"**Enhanced Prompt:** `{final_prompt}`")

        st.success("Prompt prepared successfully!")
        st.session_state.final_prompt = final_prompt  # Save for image gen

# --- Advanced Settings ---
with st.expander("⚙️ Advanced Settings"):
    num_inference_steps = st.slider("Number of Inference Steps", 1, 100, 50)
    guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)

# --- Generate Image Button ---
if st.button("Generate Image"):
    final_prompt = st.session_state.get("final_prompt", prompt_input)

    if not final_prompt:
        st.warning("Please enter or prepare a prompt first!")
    else:
        with st.spinner("Generating image..."):
            try:
                stub = get_stub()
                request = pb2.TextPrompt(
                    prompt=final_prompt,
                    context=context,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale
                )
                response = stub.GenerateImage(request)
                if response.image_path:
                    st.image(response.image_path, caption="Generated Image", use_container_width=True)
                else:
                    st.error("No image path returned.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
