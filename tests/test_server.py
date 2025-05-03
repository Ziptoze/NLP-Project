import grpc
import app.proto.text2video_pb2 as pb2
import app.proto.text2video_pb2_grpc as pb2_grpc

def test_valid_request():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = pb2_grpc.Text2VideoServiceStub(channel)
        response = stub.GenerateVideo(pb2.TextRequest(text="A cat on a skateboard", context="funny"))
        assert response.status == 200

def test_invalid_request():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = pb2_grpc.Text2VideoServiceStub(channel)
        response = stub.GenerateVideo(pb2.TextRequest(text="", context="sad"))
        assert response.status == 400
