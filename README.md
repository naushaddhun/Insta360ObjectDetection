# Insta 360 X4 environment understanding Server

A Python-based server that captures live camera frames, runs an LLM-based object-detection pipeline via [lmdeploy](https://github.com/OpenGVLab/lmdeploy), and broadcasts JSON results (with object names, bounding boxes, interactions, confidence scores and inference time) over WebSockets to any connected client.

## Features

- **Real-time frame capture** via OpenCV  
- **LLM inference** for scene understanding (uses `OpenGVLab/InternVL2_5-4B-AWQ`)  
- **JSON parsing** (with fallback) and automatic brace-balancing  
- **Per-inference timing** logged and returned  
- **WebSocket server** for easy integration with front-ends or other services  
- **WNVIDIA CUDA** Download the CUDA version12.9  

## Installation

1. **CUDA installation**  
   Download the [Cuda toolkit 12.9](https://developer.nvidia.com/cuda-downloads)

2. **Clone the repo**  
   ```bash
   git clone https://github.com/naushaddhun/Insta360ObjectDetection.git
   cd Insta360ObjectDetection

## Step 2: Create & Activate a Virtual Environment

1. **Create the venv**  
   ```bash
   python3 -m venv venv

   1. Activate it

        On Linux/macOS
        source venv/bin/activate

        On Windows (PowerShell)
        .\venv\Scripts\Activate.ps1

        On Windows (CMD)
        venv\Scripts\activate.bat
   
   2. Install Dependencies

        pip install --upgrade pip
        pip install -r requirements.txt

2. **Start the Server**
    ```bash
    python InternVLWebsocket.py

3. **Listen to the websocket**
    ```bash
    python WSListener.py