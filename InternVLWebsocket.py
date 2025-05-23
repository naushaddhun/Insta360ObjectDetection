import cv2
import torch
import time
import json
import asyncio
import websockets
import base64
import numpy as np
import os
import re
import logging
import threading
from datetime import datetime
from lmdeploy import pipeline
from lmdeploy.vl import load_image

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Starting Insta360 X4 object detection server")

# -----------------------------------------------------------------------------
# Initialize Model
# -----------------------------------------------------------------------------
model_name = 'OpenGVLab/InternVL2_5-4B-AWQ'
pipe = pipeline(model_name)
logger.info(f"Initialized model: {model_name}")

# -----------------------------------------------------------------------------
# Global Variables
# -----------------------------------------------------------------------------
analysis_in_progress = False    
connected_clients = set()  # Set of currently connected WebSocket clients

# -----------------------------------------------------------------------------
# Helper Function: parse_json_response
# Removes markdown fences and extracts the JSON substring.
# -----------------------------------------------------------------------------
def parse_json_response(response_text):
    text = response_text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        return text[start:end+1]
    return None

# -----------------------------------------------------------------------------
# Helper Function: balance_braces
# Adds missing closing braces if needed.
# -----------------------------------------------------------------------------
def balance_braces(s):
    open_braces = s.count("{")
    close_braces = s.count("}")
    if open_braces > close_braces:
        logger.info(f"Balancing braces: {open_braces} '{{' vs {close_braces} '}}'. Adding {open_braces - close_braces} missing '}}'.")
        s = s + ("}" * (open_braces - close_braces))
    return s

# -----------------------------------------------------------------------------
# Helper Function: fallback_parse
# Attempts to parse raw data if valid JSON isn't returned.
# Expected raw format per line:
# name: <object name>, x: <value>, y: <value>, z: <value>, depth: <value>, interactions: <interaction1>; <interaction2>; ...
# -----------------------------------------------------------------------------
def fallback_parse(raw_text):
    objects = []
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    pattern = (
        r"name\s*:\s*(?P<name>[^,]+),\s*"
        r"bbox\s*:\s*\[\s*(?P<xmin>[\d.]+)\s*,\s*(?P<ymin>[\d.]+)\s*,\s*"
        r"(?P<xmax>[\d.]+)\s*,\s*(?P<ymax>[\d.]+)\s*\],\s*"
        r"interactions\s*:\s*(?P<interactions>.+)"
    )
    for line in lines:
        try:
            m = re.search(pattern, line, re.IGNORECASE)
            if not m:
                continue
            obj = {
                "name": m.group("name").strip(),
                "bbox": [
                    float(m.group("xmin")),
                    float(m.group("ymin")),
                    float(m.group("xmax")),
                    float(m.group("ymax"))
                ],
                "interactions": [
                    s.strip() for s in m.group("interactions").split(";") if s.strip()
                ]
            }
            objects.append(obj)
        except Exception:
            logger.exception("Malformed object skipped during fallback parsing")
            continue
    return objects


# -----------------------------------------------------------------------------
# Function: analyze_frame
# -----------------------------------------------------------------------------
def analyze_frame(frame, prompt):
    """
    Saves the frame, loads it, and calls the LLM pipeline with the prompt.
    The prompt instructs the LLM to return valid JSON with an "objects" key,
    where each object contains "name", "x", "y", "z", "depth", and "interactions".
    If the output is truncated, the prompt asks for raw line format.
    """
    global analysis_in_progress
    try:
        analysis_in_progress = True
        logger.info("[INFO] Analysis started")

        image_path = "temp_frame.jpg"
        cv2.imwrite(image_path, frame)
        logger.info(f"[INFO] Frame saved to {image_path}")

        if not os.path.exists(image_path) or os.path.getsize(image_path) == 0:
            logger.error("Error: Failed to save frame as image")
            analysis_in_progress = False
            return None

        image = load_image(image_path)
        logger.info("[INFO] Image loaded successfully")

        gen_config = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "max_new_tokens": 256
        }
        logger.info(f"[INFO] Generation config: {gen_config}")

        start_time = time.time()
        response = pipe((prompt, image), **gen_config)
        elapsed_time = time.time() - start_time
        logger.info(f"[INFO] Inference took {elapsed_time:.2f} seconds")
        logger.info(f"[RESULT] Model response: {response.text}")

        # Clean and balance the response
        cleaned_response = parse_json_response(response.text)
        if not cleaned_response:
            logger.error("Could not extract JSON from model response")
            objects_list = []
        else:
            balanced_response = balance_braces(cleaned_response)
            last_closing_brace = balanced_response.rfind("}")
            if last_closing_brace != -1:
                balanced_response = balanced_response[:last_closing_brace + 1]
            try:
                result = json.loads(balanced_response)
                objects_list = result.get("objects", [])
            except Exception as e:
                logger.error("Error parsing JSON from model response, attempting fallback", exc_info=True)
                objects_list = fallback_parse(balanced_response)
        logger.info(f"[INFO] Parsed objects: {objects_list}")
        analysis_in_progress = False
        logger.info("[INFO] Analysis completed")
        return objects_list

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
    finally:
        analysis_in_progress = False
        logger.info("[INFO] Analysis process finalized")
    return None

# -----------------------------------------------------------------------------
# Async Function: broadcast_result
# -----------------------------------------------------------------------------
async def broadcast_result(result_json):
    for client in connected_clients.copy():
        try:
            await client.send(result_json)
        except Exception as e:
            logger.error("Error sending result to client", exc_info=True)

# -----------------------------------------------------------------------------
# WebSocket Handler
# -----------------------------------------------------------------------------
async def websocket_handler(websocket):
    client_addr = websocket.remote_address
    logger.info(f"[INFO] Client connected: {client_addr}")
    connected_clients.add(websocket)
    try:
        await websocket.send(json.dumps({
            "status": "connected",
            "message": "Server broadcasting object detection results"
        }))
        await asyncio.Future()  # run forever
    except websockets.exceptions.ConnectionClosed:
        logger.info(f"[INFO] Client disconnected: {client_addr}")
    finally:
        connected_clients.remove(websocket)

# -----------------------------------------------------------------------------
# Camera Capture and Analysis Loop (runs in a separate thread)
# -----------------------------------------------------------------------------
def camera_loop(loop):
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using DirectShow backend
    if not cap.isOpened():
        logger.error("Error: Could not open camera.")
        return

    desired_width = 2880
    desired_height = 1440
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)
    logger.info(f"Camera initialized with resolution: {desired_width}x{desired_height}")

    next_analysis_time = time.time() + 2  # Initial delay before first analysis
    last_frame = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Warning: Failed to capture frame. Retrying...")
                time.sleep(0.5)
                continue

            last_frame = frame.copy()
            current_time = time.time()
            if current_time >= next_analysis_time and not analysis_in_progress:
                prompt = (
                    "Analyze the image and identify all visible objects (excluding people). "
                    "For each object, provide its details in valid JSON format with the following fields:\n"
                    "  • \"name\" (string)\n"
                    "  • \"bbox\" (array of four numbers): [x_min, y_min, x_max, y_max], pixel coordinates of its 2D bounding box\n"
                    "  • \"interactions\" (array of strings), possible interactions with other objects\n"
                    "  • \"confidence\" (number), confidence score for the detected object\n"
                    "Return a JSON object with a key \"objects\" containing an array of these objects. "
                    "Output only valid JSON. If you cannot return full JSON due to token limits, "
                    "then output one object per line in this format:\n"
                    "  name: <object name>, bbox: [<x_min>,<y_min>,<x_max>,<y_max>], confidence: <float>, interactions: <int1>; <int2>; ..."
                )
                logger.info(f"[INFO] Starting analysis with prompt: {prompt}")
                objects_list = analyze_frame(last_frame, prompt)
                if objects_list is not None:
                    result = {
                        "timestamp": time.time(),
                        "objects": objects_list,
                        "scene_understanding": {
                            "surfaces": [],
                            "spatial_anchors": []
                        }
                    }
                    result_json = json.dumps(result)
                    logger.info(f"[INFO] Broadcasting result: {result_json}")
                    asyncio.run_coroutine_threadsafe(broadcast_result(result_json), loop)
                else:
                    logger.error("Analysis returned no objects")
                next_analysis_time = current_time + 10  # Analysis interval
            time.sleep(0.1)  # Lower CPU usage
    finally:
        cap.release()
        logger.info("[INFO] Camera released")

# -----------------------------------------------------------------------------
# Main: Start the WebSocket server and camera loop
# -----------------------------------------------------------------------------
async def async_main():
    loop = asyncio.get_running_loop()
    cam_thread = threading.Thread(target=camera_loop, args=(loop,), daemon=True)
    cam_thread.start()
    logger.info("[INFO] Camera loop started")
    server = await websockets.serve(websocket_handler, "0.0.0.0", 8765)
    logger.info("[INFO] WebSocket server started on ws://0.0.0.0:8765")
    await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(async_main())