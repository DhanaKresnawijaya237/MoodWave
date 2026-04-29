# MoodWave — TouchDesigner Setup (Pre-loaded Timeline Flow)

## What changed

Old flow (removed):
- Frontend sent one mood chunk every ~100ms during playback
- Frontend sent `/moodwave/time` pings so TD could keep UI in sync

New flow:
- Backend analyzes the full song, saves it to `backend/uploads/current.wav`
- Frontend sends **one timeline payload** to the bridge after analysis finishes
- Frontend sends **play / pause / seek** only (no time sync)
- TD owns the audio playhead and reads the timeline arrays itself

## OSC addresses TD receives

| Address                              | Type                | Purpose                                |
|--------------------------------------|---------------------|----------------------------------------|
| `/moodwave/filepath`                 | string              | Absolute path to `current.wav`         |
| `/moodwave/duration`                 | float               | Song duration in seconds               |
| `/moodwave/chunk_duration`           | float               | Seconds per chunk (10)                 |
| `/moodwave/num_chunks`               | int                 | Number of chunks in the timeline       |
| `/moodwave/timeline/valence`         | float[]             | Valence per chunk                      |
| `/moodwave/timeline/arousal`         | float[]             | Arousal per chunk                      |
| `/moodwave/timeline/tempo`           | float[]             | Tempo per chunk                        |
| `/moodwave/timeline/energy`          | float[]             | Energy per chunk                       |
| `/moodwave/timeline/brightness`      | float[]             | Brightness per chunk                   |
| `/moodwave/timeline/dominant`        | int[]               | Dominant mood index (0..5) per chunk   |
| `/moodwave/timeline/mood/{name}`     | float[]             | Probability per chunk for one mood     |
| `/moodwave/transport`                | 0 / 1               | Pause / Play                           |
| `/moodwave/seek`                     | float               | Seek to time (seconds)                 |

Mood names: `energetic`, `happy`, `calm`, `romantic`, `sad`, `angry`.

---

## Required operators

### 1. OSC In DAT — `oscin_ctrl`
- **Network Port:** 7000
- **Active:** on
- **Callbacks DAT:** create a Text DAT (e.g. `osc_callbacks`) and point the
  OSC In DAT's Callbacks parameter at it.

Paste this into `osc_callbacks`:

```python
def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    audio = op('audiofilein1')

    if address == '/moodwave/filepath' and len(args) > 0:
        filepath = args[0]
        print(f'[MoodWave] Loading file: {filepath}')
        audio.par.file.val = filepath
        audio.par.cuepoint.val = 0
        audio.par.cuepulse.pulse()

    elif address == '/moodwave/transport' and len(args) > 0:
        audio.par.play = int(args[0])
        print(f"[MoodWave] Transport: {'play' if int(args[0]) else 'pause'}")

    elif address == '/moodwave/seek' and len(args) > 0:
        t = float(args[0])
        audio.par.cuepoint.val = t
        audio.par.cuepulse.pulse()
        print(f'[MoodWave] Seek -> {t:.2f}s')
```

### 2. OSC In CHOP — `oscin_timeline`
- **Network Port:** 7000 (same port is fine — TD splits between DAT and CHOP)
- **Address Scope:** `/moodwave/timeline/*`
- This CHOP will expose multi-sample channels, one per address. Each channel's
  sample count = number of chunks. Channel names appear as
  `moodwave/timeline/valence`, etc.

> If you want to keep DAT and CHOP cleanly separated, use two OSC In operators
> on the same port — TD allows multiple receivers to listen to the same port.

### 3. Audio File In CHOP — `audiofilein1`
- **File:** (leave blank — it gets set by OSC)
- **Play:** off by default (transport OSC flips it on)
- **Cue:** used by the seek handler

### 4. Current-chunk lookup — Script CHOP `current_mood`
Create a Script CHOP, paste into its `onCook`:

```python
import colorsys

def onCook(scriptOp):
    scriptOp.clear()
    scriptOp.numSamples = 1

    audio    = op('audiofilein1')
    timeline = op('oscin_timeline')

    # Current audio time in seconds
    t = float(audio['t'][0]) if audio is not None else 0.0

    chunk_dur = 10.0
    chunk_idx = int(t // chunk_dur)

    val_ch = timeline['moodwave/timeline/valence']
    aro_ch = timeline['moodwave/timeline/arousal']

    # Bail out cleanly until the timeline has arrived
    if val_ch is None or aro_ch is None or val_ch.numSamples == 0:
        scriptOp.appendChan('valence')[0] = 0
        scriptOp.appendChan('arousal')[0] = 0
        scriptOp.appendChan('r')[0] = 0.5
        scriptOp.appendChan('g')[0] = 0.5
        scriptOp.appendChan('b')[0] = 0.5
        return

    n = val_ch.numSamples
    chunk_idx = max(0, min(chunk_idx, n - 1))

    valence = float(val_ch[chunk_idx])
    arousal = float(aro_ch[chunk_idx])

    # HSV -> RGB mapping (same as before)
    u = (valence + 1) / 2
    hue = (220 + u * (55 - 220)) / 360
    sat = 0.45 + ((arousal + 1) / 2) * 0.55
    bri = 0.65 + ((abs(valence) + abs(arousal)) / 2) * 0.35
    r, g, b = colorsys.hsv_to_rgb(hue, sat, bri)

    scriptOp.appendChan('valence')[0] = valence
    scriptOp.appendChan('arousal')[0] = arousal
    scriptOp.appendChan('r')[0] = r
    scriptOp.appendChan('g')[0] = g
    scriptOp.appendChan('b')[0] = b
```

This CHOP now outputs `valence`, `arousal`, `r`, `g`, `b` for the **currently
playing chunk**. Reference any of these from the rest of your visuals.

### 5. MJPEG server (unchanged)
Keep using `td_mjpeg_server.py`. Make sure:
- `TOP_PATH` in that script still points at your output TOP
- Execute DAT is set up to call `update_frame()` on `onFrameStart`

---

## Minimum network sketch

```
OSC In DAT (oscin_ctrl) -> osc_callbacks Text DAT
                              |
                              v
                         Audio File In CHOP (audiofilein1)
                              |
                              | t (playhead)
                              v
OSC In CHOP (oscin_timeline) -> Script CHOP (current_mood) -> your visuals
                              (timeline arrays)
```

---

## Startup order

1. `python osc_bridge.py`            (in `touch_designer/`)
2. `uvicorn main:app --reload --port 8000`  (in `backend/`)
3. Open the TouchDesigner project — watch Textport for
   `[MoodWave] MJPEG server running...`
4. Open `frontend/moodwave.html` in a browser and drop an audio file in

On upload, Textport should show:
- `[MoodWave] Loading file: C:/.../uploads/current.wav`
- `[MoodWave] Transport: play`

Once the timeline payload has been received, the `current_mood` Script CHOP
will start tracking the audio's playhead with no extra plumbing.
