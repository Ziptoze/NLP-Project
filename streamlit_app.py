import streamlit as st
import grpc

import app.text2image_pb2 as pb2
import app.text2image_pb2_grpc as pb2_grpc
import asyncio

st.title("Text2Image Generator (Stable Diffusion v2.1)")

prompt = st.text_input("Enter your image prompt:")

if st.button("Generate Image"):
    async def generate():
        async with grpc.aio.insecure_channel('localhost:50051') as channel:
            stub = pb2_grpc.Text2ImageServiceStub(channel)
            response = await stub.GenerateImage(pb2.TextPrompt(prompt=prompt))
            st.image(response.image_path)

    asyncio.run(generate())
