import asyncio
import websockets
import logging

logging.basicConfig(level=logging.INFO)

async def listen():
    # Replace with a valid WebSocket URI
    uri = "ws://localhost:8765"  
    try:
        async with websockets.connect(uri) as websocket:
            logging.info(f"Connected to {uri}")
            while True:
                message = await websocket.recv()
                logging.info(f"Received message: {message}")
    except websockets.ConnectionClosed as e:
        logging.warning(f"Connection closed: {e}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(listen())
