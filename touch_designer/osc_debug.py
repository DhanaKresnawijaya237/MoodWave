"""
MoodWave OSC Debug Suite
Run this from your backend folder to diagnose exactly where the OSC pipeline breaks.

Usage:
    python osc_debug.py
"""

import sys
import socket
import time
import threading
import json

print("=" * 60)
print("  MoodWave OSC Debug Suite")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# TEST 1: Check python-osc is installed
# ─────────────────────────────────────────────────────────────
print("\n[TEST 1] Checking python-osc installation...")
try:
    from pythonosc import udp_client, dispatcher, osc_server
    print("  [OK] python-osc is installed")
except ImportError:
    print("  [FAIL] python-osc is NOT installed")
    print("         Fix: pip install python-osc")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────
# TEST 2: Check websockets is installed
# ─────────────────────────────────────────────────────────────
print("\n[TEST 2] Checking websockets installation...")
try:
    import websockets
    print("  [OK] websockets is installed")
except ImportError:
    print("  [FAIL] websockets is NOT installed")
    print("         Fix: pip install websockets")

# ─────────────────────────────────────────────────────────────
# TEST 3: Send a raw OSC packet to TD and confirm UDP reaches port 7000
# ─────────────────────────────────────────────────────────────
print("\n[TEST 3] Sending test OSC message to 127.0.0.1:7000 ...")
try:
    client = udp_client.SimpleUDPClient("127.0.0.1", 7000)
    client.send_message("/moodwave/test", 1.0)
    client.send_message("/moodwave/valence", 0.5)
    client.send_message("/moodwave/arousal", 0.3)
    client.send_message("/moodwave/tempo", 120.0)
    client.send_message("/moodwave/energy", 0.04)
    client.send_message("/moodwave/brightness", 0.6)
    for mood, val in [("energetic",0.1),("happy",0.4),("calm",0.3),("romantic",0.1),("sad",0.05),("angry",0.05)]:
        client.send_message(f"/moodwave/mood/{mood}", float(val))
    client.send_message("/moodwave/dominant_idx", 1)
    print("  [OK] OSC packets sent (UDP is fire-and-forget — no confirmation possible)")
    print("       >>> CHECK TOUCHDESIGNER NOW <<<")
    print("       In TD: OSC In CHOP → Network Port = 7000, Active = On")
    print("       You should see channels: /moodwave/test, /moodwave/valence, etc.")
except Exception as e:
    print(f"  [FAIL] Could not send OSC: {e}")

# ─────────────────────────────────────────────────────────────
# TEST 4: Check if port 7001 (WebSocket bridge) is already in use
# ─────────────────────────────────────────────────────────────
print("\n[TEST 4] Checking if WebSocket bridge port 7001 is free...")
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(1)
result = sock.connect_ex(('127.0.0.1', 7001))
sock.close()
if result == 0:
    print("  [WARN] Port 7001 is already IN USE — another process may be running")
    print("         Kill the old bridge process and re-run osc_bridge.py")
else:
    print("  [OK] Port 7001 is free — bridge can start on it")

# ─────────────────────────────────────────────────────────────
# TEST 5: Spam OSC every second for 10 seconds so you can watch TD
# ─────────────────────────────────────────────────────────────
print("\n[TEST 5] Sending OSC bursts every 1s for 10 seconds...")
print("         Watch your OSC In CHOP in TouchDesigner for channel activity.")
print("         Press Ctrl+C to stop early.\n")

moods = ["energetic", "happy", "calm", "romantic", "sad", "angry"]
try:
    for i in range(10):
        t = i / 9.0  # 0 → 1
        valence  = round(-1.0 + t * 2.0, 3)   # sweeps -1 → +1
        arousal  = round(0.5 * (1 + __import__('math').sin(t * 3.14)), 3)
        dominant_idx = i % 6
        dominant = moods[dominant_idx]

        client.send_message("/moodwave/valence",    valence)
        client.send_message("/moodwave/arousal",    arousal)
        client.send_message("/moodwave/tempo",      float(80 + i * 10))
        client.send_message("/moodwave/energy",     float(0.01 + t * 0.09))
        client.send_message("/moodwave/brightness", float(t))
        client.send_message("/moodwave/dominant_idx", dominant_idx)
        client.send_message("/moodwave/dominant_str", dominant)
        for j, mood in enumerate(moods):
            val = 1.0 if j == dominant_idx else 0.0
            client.send_message(f"/moodwave/mood/{mood}", val)

        print(f"  Burst {i+1}/10 → valence={valence:+.2f}  arousal={arousal:.2f}  dominant={dominant}")
        time.sleep(1.0)

    print("\n[DONE] If you saw channels moving in TD, OSC is working!")
    print("       If nothing moved, see the checklist below.")

except KeyboardInterrupt:
    print("\n  Stopped by user.")

# ─────────────────────────────────────────────────────────────
# Checklist
# ─────────────────────────────────────────────────────────────
print("""
─────────────────────────────────────────────────────────────
  TouchDesigner Checklist (if Test 5 bursts showed nothing)
─────────────────────────────────────────────────────────────

  1. OSC In CHOP settings:
       Network Port  = 7000        (must match TD_PORT in bridge)
       Active        = On          (toggle it off then on again)
       Local Address = (leave blank, or 127.0.0.1)
       Protocol      = UDP         (NOT TCP)

  2. After changing port/active, right-click the CHOP → "Pulse" (cook it)

  3. Windows Firewall:
       TouchDesigner must be allowed on private networks.
       Open: Windows Security → Firewall → Allow an app → check TouchDesigner

  4. If TD is on a different machine, change TD_HOST in osc_bridge.py
     from "127.0.0.1" to the machine's local IP (e.g. 192.168.1.x)

  5. Confirm channels appear: click the OSC In CHOP, look at the
     Info CHOP panel — "Active Channels" should be > 0 after a burst.

  6. Try toggling the OSC In CHOP Active parameter Off → On
     while the bursts are running.
─────────────────────────────────────────────────────────────
""")