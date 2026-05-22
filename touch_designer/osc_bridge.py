"""
MoodWave OSC Bridge
Receives a full mood timeline from the browser via WebSocket and forwards it
as OSC messages to TouchDesigner.

New flow (pre-loaded, not streaming):
  1. Frontend sends ONE 'timeline' payload after analysis completes.
     The payload contains: filepath, duration, chunk_duration, and arrays of
     valence/arousal/tempo/energy/brightness and per-mood distribution arrays.
  2. Frontend sends transport messages (play, pause, seek) when the user
     interacts with the player UI.

Setup:
  pip install python-osc websockets

Run BEFORE opening the browser:
  python osc_bridge.py

Then open http://localhost:8000 (or the frontend HTML) and load a song.
"""

import asyncio
import json
import websockets
from pythonosc import udp_client

# -- Config ------------------------------------------------------------------
TD_HOST  = "127.0.0.1"   # TouchDesigner machine IP (127.0.0.1 = same computer)
TD_PORT  = 7000          # Must match OSC In CHOP/DAT "Network Port" in TD
WS_PORT  = 7011          # WebSocket port - browser connects here
# ----------------------------------------------------------------------------

MOODS = ["energetic", "happy", "calm", "romantic", "sad", "angry"]

osc = udp_client.SimpleUDPClient(TD_HOST, TD_PORT)
print(f"[Bridge] OSC -> {TD_HOST}:{TD_PORT}")
print(f"[Bridge] WebSocket listening on ws://localhost:{WS_PORT}")
print(f"[Bridge] Waiting for browser to connect...\n")


def send_timeline(payload):
    """
    Forward a full timeline payload to TD as OSC messages.

    Expected payload shape (all arrays same length = num chunks):
      {
        "filepath": "C:/.../uploads/current.wav",
        "duration": 183.2,
        "chunk_duration": 10,
        "num_chunks": 18,
        "valence":    [float, ...],
        "arousal":    [float, ...],
        "tempo":      [float, ...],
        "energy":     [float, ...],
        "brightness": [float, ...],
        "dominant_idx": [int, ...],
        "mood": { "energetic": [float...], "happy": [...], ... },
        "stems": {
          "vocals": "C:/.../stems/vocals.wav",
          "drums": "C:/.../stems/drums.wav",
          "bass": "C:/.../stems/bass.wav",
          "guitar": "C:/.../stems/guitar.wav",
          "piano": "C:/.../stems/piano.wav",
          "other": "C:/.../stems/other.wav"
        }
      }
    """
    filepath       = str(payload.get("filepath", ""))
    duration       = float(payload.get("duration", 0))
    chunk_duration = float(payload.get("chunk_duration", 10))
    num_chunks     = int(payload.get("num_chunks", 0))

    # 1. File path + meta (sent first so TD can load audio before timeline data)
    if filepath:
        osc.send_message("/moodwave/filepath", filepath)
    osc.send_message("/moodwave/duration",       duration)
    osc.send_message("/moodwave/chunk_duration", chunk_duration)
    osc.send_message("/moodwave/num_chunks",     num_chunks)

    # 1b. Stem file paths (if Demucs separation was successful)
    stems = payload.get("stems")
    if stems and isinstance(stems, dict):
        for stem_name, stem_path in stems.items():
            if stem_path:
                osc.send_message(f"/moodwave/stems/{stem_name}", str(stem_path))
        print(f"[OSC] STEMS       {list(stems.keys())}")

    # 2. Timeline arrays - each becomes a multi-sample channel in TD's OSC In CHOP
    def _send_array(addr, arr):
        if not arr:
            return
        # Coerce to floats to avoid int/float type mismatches
        osc.send_message(addr, [float(x) for x in arr])

    _send_array("/moodwave/timeline/valence",    payload.get("valence"))
    _send_array("/moodwave/timeline/arousal",    payload.get("arousal"))
    _send_array("/moodwave/timeline/tempo",      payload.get("tempo"))
    _send_array("/moodwave/timeline/energy",     payload.get("energy"))
    _send_array("/moodwave/timeline/brightness", payload.get("brightness"))
    _send_array("/moodwave/timeline/dominant",   payload.get("dominant_idx"))

    mood = payload.get("mood") or {}
    for name in MOODS:
        _send_array(f"/moodwave/timeline/mood/{name}", mood.get(name))

    print(f"[OSC] TIMELINE  file={filepath}")
    print(f"              duration={duration:.2f}s  chunks={num_chunks}")


def send_transport(msg_type, value):
    """Transport control: play, pause, seek."""
    if msg_type == "play":
        osc.send_message("/moodwave/transport", 1)
        osc.send_message("/moodwave/seek", float(value))
        print(f"[OSC] PLAY  @ {value:.2f}s")
    elif msg_type == "pause":
        osc.send_message("/moodwave/transport", 0)
        print(f"[OSC] PAUSE @ {value:.2f}s")
    elif msg_type == "seek":
        osc.send_message("/moodwave/seek", float(value))
        print(f"[OSC] SEEK  -> {value:.2f}s")


async def handle_client(websocket):
    addr = websocket.remote_address
    print(f"[Bridge] Browser connected from {addr}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                msg_type = data.get("type")

                if msg_type == "timeline":
                    send_timeline(data)

                elif msg_type in ("play", "pause", "seek"):
                    send_transport(msg_type, data.get("value", 0))

                else:
                    print(f"[Bridge] Unknown message type: {msg_type!r}")

            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Bridge] Parse error: {e}")

    except websockets.exceptions.ConnectionClosed:
        print(f"[Bridge] Browser disconnected from {addr}")


async def main():
    async with websockets.serve(handle_client, "localhost", WS_PORT):
        print(f"[Bridge] Running. Press Ctrl+C to stop.\n")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[Bridge] Stopped.")
