"""
MoodWave OSC Bridge
Receives mood data from the browser via WebSocket and forwards it as OSC to TouchDesigner.

Setup:
  pip install python-osc websockets

Run BEFORE opening the browser:
  python osc_bridge.py

Then open http://localhost:8000 and load a song.
"""

import asyncio
import json
import websockets
from pythonosc import udp_client

# ── Config ─────────────────────────────────────────────────────
TD_HOST  = "127.0.0.1"   # TouchDesigner machine IP (127.0.0.1 = same computer)
TD_PORT  = 7000           # Must match OSC In CHOP "Network Port" in TouchDesigner
WS_PORT  = 7001           # WebSocket port — browser connects here
# ───────────────────────────────────────────────────────────────

MOODS = ["energetic", "happy", "calm", "romantic", "sad", "angry"]

osc = udp_client.SimpleUDPClient(TD_HOST, TD_PORT)
print(f"[Bridge] OSC → {TD_HOST}:{TD_PORT}")
print(f"[Bridge] WebSocket listening on ws://localhost:{WS_PORT}")
print(f"[Bridge] Waiting for browser to connect...\n")


async def handle_client(websocket):
    addr = websocket.remote_address
    print(f"[Bridge] Browser connected from {addr}")

    try:
        async for message in websocket:
            try:
                data = json.loads(message)

                # Transport control messages (play/pause/seek/time)
                msg_type = data.get("type")
                if msg_type in ("play", "pause", "seek", "time"):
                    value = float(data.get("value", 0))
                    if msg_type == "play":
                        osc.send_message("/moodwave/transport", 1)
                        osc.send_message("/moodwave/seek", value)
                        print(f"[OSC] PLAY  @ {value:.2f}s")
                    elif msg_type == "pause":
                        osc.send_message("/moodwave/transport", 0)
                        print(f"[OSC] PAUSE @ {value:.2f}s")
                    elif msg_type == "seek":
                        osc.send_message("/moodwave/seek", value)
                        print(f"[OSC] SEEK  → {value:.2f}s")
                    elif msg_type == "time":
                        osc.send_message("/moodwave/time", value)
                    continue

                # Mood chunk data
                valence    = float(data.get("valence", 0))
                arousal    = float(data.get("arousal", 0))
                tempo      = float(data.get("tempo", 120))
                energy     = float(data.get("energy", 0))
                brightness = float(data.get("brightness", 0))
                dominant   = str(data.get("dominant", "calm"))
                dist       = data.get("distribution", {})

                osc.send_message("/moodwave/valence",    valence)
                osc.send_message("/moodwave/arousal",    arousal)
                osc.send_message("/moodwave/tempo",      tempo)
                osc.send_message("/moodwave/energy",     energy)
                osc.send_message("/moodwave/brightness", brightness)

                idx = MOODS.index(dominant) if dominant in MOODS else 0
                osc.send_message("/moodwave/dominant_idx", idx)
                osc.send_message("/moodwave/dominant_str", dominant)

                for mood in MOODS:
                    osc.send_message(f"/moodwave/mood/{mood}", float(dist.get(mood, 0)))

                print(f"[OSC] {dominant:10s} | V={valence:+.3f}  A={arousal:+.3f}  "
                      f"BPM={tempo:.0f}  E={energy:.4f}")

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