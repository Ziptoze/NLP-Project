# Text2Video Microservice

## Setup Instructions
```bash
pip install -r requirements.txt
grpc_tools.protoc -I. --python_out=. --grpc_python_out=. app/proto/text2video.proto
python app/server.py
```

## Test with Postman
Use grpc extension for Postman. Send a request to `localhost:50051/Text2VideoService/GenerateVideo` with:
```json
{
  "text": "A dragon flying through a storm",
  "context": "epic"
}
```

## Deployment
Use Docker:
```bash
docker build -t text2video .
docker run -p 50051:50051 text2video
```

## Frontend
```bash
streamlit run frontend/app.py
```

## Notes
- Model: sd-turbo (lightweight Stable Diffusion)
- Generated frames only (you can later animate frames into a video using OpenCV)
