import asyncio
import grpc
import time
import matplotlib.pyplot as plt
import text2image_pb2 as pb2
import text2image_pb2_grpc as pb2_grpc

# Server details
HOST = '127.0.0.1:8501'

# Define parameter combinations to test
INFERENCE_STEPS = [10, 20, 30]
GUIDANCE_SCALES = [5.0, 7.5, 10.0]

# Generate a single request payload
def create_prompt(num_inference_steps, guidance_scale):
    return pb2.TextPrompt(
        prompt="a sunny beach with palm trees",
        context="photorealistic",
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale
    )

# Send one request and measure response time
async def measure_response_time(stub, prompt):
    start = time.perf_counter()
    try:
        response = await stub.GenerateImage(prompt, timeout=1000)  # generous timeout
        # Optional: validate image path or response fields
        return time.perf_counter() - start
    except grpc.aio.AioRpcError as e:
        print(f"Error: {e}")
        return None

# Main evaluation loop
async def main():
    results = []

    async with grpc.aio.insecure_channel(HOST) as channel:
        stub = pb2_grpc.Text2ImageServiceStub(channel)

        for steps in INFERENCE_STEPS:
            for scale in GUIDANCE_SCALES:
                print(f"Testing: Steps={steps}, Guidance={scale}...")
                prompt = create_prompt(steps, scale)
                duration = await measure_response_time(stub, prompt)
                if duration is not None:
                    results.append((steps, scale, duration))
                    print(f"Completed in {duration:.2f} seconds.\n")
                else:
                    results.append((steps, scale, None))
                    print("Request failed.\n")

    # Display results
    print("\n=== Performance Results ===")
    for steps, scale, duration in results:
        result_str = f"{duration:.2f}s" if duration else "Failed"
        print(f"Inference Steps: {steps}, Guidance Scale: {scale} => {result_str}")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))
    for scale in GUIDANCE_SCALES:
        xs = [steps for steps, s, _ in results if s == scale]
        ys = [duration for steps, s, duration in results if s == scale]
        ax.plot(xs, ys, marker='o', label=f'Guidance {scale}')
    
    ax.set_title("Image Generation Time vs Inference Steps")
    ax.set_xlabel("Inference Steps")
    ax.set_ylabel("Response Time (s)")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    asyncio.run(main())
