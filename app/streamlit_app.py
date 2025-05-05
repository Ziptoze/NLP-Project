# streamlit_app.py

import streamlit as st
import grpc
import text2image_pb2 as pb2
import text2image_pb2_grpc as pb2_grpc
import enhancement
import sys
from speech import recognize_voice
import tempfile
import base64

# Fix torch.classes import issues
if "torch.classes" in sys.modules:
    del sys.modules["torch.classes"]

# Load enhancement models
@st.cache_resource
def load_models():
    enhancement.download_models()

load_models()

def get_stub():
    channel = grpc.insecure_channel('localhost:50051')
    return pb2_grpc.Text2ImageServiceStub(channel)

st.title("🎨 AI Image Generator")

# --- Tabs for modes ---
tab1, tab2 = st.tabs(["📝 Text2Image", "🖼️ Img2Img"])

# -------------------------------
# --- TEXT TO IMAGE TAB ---------
# -------------------------------
with tab1:
    lang = st.selectbox("Prompt Language", options=["English", "Urdu"])

    prompt_input = st.text_input("Enter your main prompt:", "")
    context = st.text_area("Enter additional context (optional):", "")
    enhance_prompt = st.checkbox("✨ Enhance Prompt")

    if "voice_prompt" not in st.session_state:
        st.session_state.voice_prompt = ""

    if st.button("🎙 Record Prompt (Voice)", key="record_voice"):
        with st.spinner("Listening..."):
            st.session_state.voice_prompt = recognize_voice(duration=5)

    st.text_area("Recognized Prompt", st.session_state.voice_prompt, key="voice_prompt_display", height=100)

    translated_prompt = ""
    enhanced_prompt = ""
    final_prompt = prompt_input or st.session_state.voice_prompt

    if st.button("Translate / Enhance Prompt", key="translate_enhance"):
        if final_prompt.strip():
            if lang == "Urdu":
                translated_prompt = enhancement.translate_to_english(final_prompt)
                st.markdown(f"**🈶 Translated Prompt:** `{translated_prompt}`")
            else:
                translated_prompt = final_prompt
                st.markdown(f"**✍️ Original Prompt Used (English):** `{translated_prompt}`")

            if enhance_prompt:
                enhanced_prompt = enhancement.enhance_prompt(translated_prompt)
                st.markdown(f"**🚀 Enhanced Prompt:** `{enhanced_prompt}`")
                st.session_state.final_prompt = enhanced_prompt
            else:
                st.session_state.final_prompt = translated_prompt

            st.success("Prompt is ready!")
        else:
            st.warning("Please enter or record a prompt first!")

    with st.expander("⚙️ Advanced Settings (Text2Image)"):
        num_inference_steps = st.slider("Number of Inference Steps", 1, 100, 50)
        guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5)

    if st.button("Generate Image", key="generate_text"):
        final_prompt = st.session_state.get("final_prompt", "")
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
                        st.image(response.image_path, caption="🖼 Generated Image", use_container_width=True)
                    else:
                        st.error("No image path returned.")
                except Exception as e:
                    st.error(f"Error generating image: {str(e)}")

# -------------------------------
# --- IMAGE TO IMAGE TAB --------
# -------------------------------
with tab2:
    uploaded_file = st.file_uploader("Upload an image to transform", type=["jpg", "jpeg", "png"])

    # Extended styles
    available_styles = [
        "Ghibli", "Anime", "Realistic", "Colorize", "Cyberpunk", "Fantasy", "Van Gogh", "Watercolor",
        "Oil Painting", "Comic Book", "3D Render", "Sketch", "Synthwave", "Pixel Art",
        "Surrealism", "Steampunk", "Noir"
    ]
    style = st.selectbox("Choose a style", options=available_styles)

    prompt = st.text_input("Optional prompt to describe the image:", "")

    with st.expander("⚙️ Advanced Settings (Img2Img)"):
        num_inference_steps = st.slider("Steps", 1, 100, 50, key="steps_img2img")
        guidance_scale = st.slider("Guidance", 1.0, 20.0, 7.5, key="guidance_img2img")
        strength = st.slider("Transformation Strength", 0.1, 1.0, 0.75, key="strength_img2img")

    if uploaded_file and st.button("Transform Image"):
        with st.spinner("Generating styled image..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_file.read())
                    image_path = tmp.name

                stub = get_stub()

                request = pb2.ImageToImagePrompt(
                    init_image_path=image_path,
                    style=style,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength
                )
                response = stub.GenerateImageFromImage(request)

                if response.image_path:
                    st.image(response.image_path, caption=f"🖼 Stylized Image: {style}", use_container_width=True)
                else:
                    st.error("No image was returned.")
            except Exception as e:
                st.error(f"Error: {str(e)}")
