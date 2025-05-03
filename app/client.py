import grpc
import app.text2image_pb2 as pb2
import app.text2image_pb2_grpc as pb2_grpc
import asyncio

async def run():
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        stub = pb2_grpc.Text2ImageServiceStub(channel)
        prompt = input("Enter prompt: ")
        response = await stub.GenerateImage(pb2.TextPrompt(prompt=prompt))
        print(f"Image generated at: {response.image_path}")

if __name__ == "__main__":
    asyncio.run(run())
