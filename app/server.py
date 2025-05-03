from concurrent import futures
import grpc
import text2image_pb2 as pb2
import text2image_pb2_grpc as pb2_grpc
from utils import generate_image

import asyncio

class Text2ImageService(pb2_grpc.Text2ImageServiceServicer):
    async def GenerateImage(self, request, context):
        try:
            image_path = generate_image(
                prompt=request.prompt,
                context=request.context,
                num_inference_steps=request.num_inference_steps or 50,
                guidance_scale=request.guidance_scale or 7.5,
            )
            return pb2.ImageResponse(image_path=image_path)
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            return pb2.ImageResponse()

async def serve():
    server = grpc.aio.server()
    pb2_grpc.add_Text2ImageServiceServicer_to_server(Text2ImageService(), server)
    server.add_insecure_port('[::]:50051')
    await server.start()
    print("gRPC server started at port 50051")
    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(serve())
