import asyncio
import websockets
import pyaudio
import json

# Audio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

async def stream_mic():
    uri = "ws://127.0.0.1:8000/ws"
    
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Connecting to Aura's Sensory Gate...")
    
    async with websockets.connect(uri) as websocket:
        print("Connected! Start speaking...")
        
        try:
            while True:
                # Read audio chunk from mic
                data = stream.read(CHUNK, exception_on_overflow=False)
                # Send raw bytes to FastAPI
                await websocket.send(data)
                
                # Check for incoming responses asynchronously
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    
                    try:
                        parsed = json.loads(message)
                        if parsed.get('type') == 'chat_stream_chunk':
                            print(parsed.get('chunk', ''), end='', flush=True)
                        elif parsed.get('type') in ('chat_stream_start', 'chat_stream_end', 'auth_success', 'pong'):
                            pass # suppress noise
                        else:
                            print(f"\n[Aura UI]: {parsed}")
                    except json.JSONDecodeError:
                        print(f"\n[Aura Message]: {message}")

                except asyncio.TimeoutError:
                    pass
                    
        except KeyboardInterrupt:
            print("\nStopping stream.")
        finally:
            stream.stop_stream()
            stream.close()
            audio.terminate()

if __name__ == "__main__":
    asyncio.run(stream_mic())
