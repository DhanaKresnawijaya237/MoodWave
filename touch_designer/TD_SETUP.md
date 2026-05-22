# MoodWave — TouchDesigner Setup (Pre-loaded Timeline Flow)

## What changed

Old flow (removed):
- Frontend sent one mood chunk every ~100ms during playback
- Frontend sent `/moodwave/time` pings so TD could keep UI in sync

New flow:
- Backend analyzes the full song, saves it as a real PCM WAV under
  `backend/uploads/run_<id>/current.wav`
- Frontend sends **one timeline payload** to the bridge after analysis finishes
- Frontend sends **play / pause / seek** only (no time sync)
- TD owns the audio playhead and reads the timeline arrays itself
- If Demucs succeeds, six stems are sent as 44.1 kHz stereo PCM WAV files:
  `vocals`, `drums`, `bass`, `guitar`, `piano`, `other`.

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
| `/moodwave/stems/vocals`             | string              | Absolute path to vocals stem           |
| `/moodwave/stems/drums`              | string              | Absolute path to drums stem            |
| `/moodwave/stems/bass`               | string              | Absolute path to bass stem             |
| `/moodwave/stems/guitar`             | string              | Absolute path to guitar stem           |
| `/moodwave/stems/piano`              | string              | Absolute path to piano stem            |
| `/moodwave/stems/other`              | string              | Absolute path to residual other stem   |
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
STEM_AUDIO_OPS = [
    'audiofilein_vocals',
    'audiofilein_drums',
    'audiofilein_bass',
    'audiofilein_guitar',
    'audiofilein_piano',
    'audiofilein_other',
]

def _audio_ops():
    ops = [op('audiofilein1')]
    ops.extend(op(name) for name in STEM_AUDIO_OPS)
    return [audio for audio in ops if audio is not None]

def _load_audio(audio, filepath):
    audio.par.file.val = filepath
    audio.par.cuepoint.val = 0
    audio.par.cuepulse.pulse()

def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    audio = op('audiofilein1')

    if address == '/moodwave/filepath' and len(args) > 0:
        filepath = args[0]
        print(f'[MoodWave] Loading file: {filepath}')
        if audio:
            _load_audio(audio, filepath)

    elif address == '/moodwave/transport' and len(args) > 0:
        play = int(args[0])
        for audio in _audio_ops():
            audio.par.play = play
        print(f"[MoodWave] Transport: {'play' if play else 'pause'}")

    elif address == '/moodwave/seek' and len(args) > 0:
        t = float(args[0])
        for audio in _audio_ops():
            audio.par.cuepoint.val = t
            audio.par.cuepulse.pulse()
        print(f'[MoodWave] Seek -> {t:.2f}s')

    elif address.startswith('/moodwave/stems/') and len(args) > 0:
        stem_name = address.split('/')[-1]
        stem = op(f'audiofilein_{stem_name}')
        if stem:
            print(f'[MoodWave] Loading stem {stem_name}: {args[0]}')
            _load_audio(stem, args[0])
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

> **Stem support:** If Demucs separation succeeds, six additional
> `/moodwave/stems/*` paths are sent. You can create extra Audio File In CHOPs
> named exactly:
>
> - `audiofilein_vocals`
> - `audiofilein_drums`
> - `audiofilein_bass`
> - `audiofilein_guitar`
> - `audiofilein_piano`
> - `audiofilein_other`
>
> The callback above auto-loads any `/moodwave/stems/{name}` path into
> `audiofilein_{name}` and keeps all stem players synced with the main audio.
> If you prefer explicit branches, the equivalent callback checks are:
>
> ```python
> elif address == '/moodwave/stems/vocals' and len(args) > 0:
>     op('audiofilein_vocals').par.file.val = args[0]
> elif address == '/moodwave/stems/drums' and len(args) > 0:
>     op('audiofilein_drums').par.file.val = args[0]
> elif address == '/moodwave/stems/bass' and len(args) > 0:
>     op('audiofilein_bass').par.file.val = args[0]
> elif address == '/moodwave/stems/guitar' and len(args) > 0:
>     op('audiofilein_guitar').par.file.val = args[0]
> elif address == '/moodwave/stems/piano' and len(args) > 0:
>     op('audiofilein_piano').par.file.val = args[0]
> elif address == '/moodwave/stems/other' and len(args) > 0:
>     op('audiofilein_other').par.file.val = args[0]
> ```
>
> Wire all Audio File In CHOPs to the same transport/seek logic (or group them
> in a Container COMP) so they stay in sync.

### 3b. Stem control CHOPs for visuals

For instrument-reactive visuals, create a small control branch per stem:

```text
audiofilein_vocals -> analyze_vocals -> math_vocals -> lag_vocals -> null_vocals_ctrl
audiofilein_drums  -> analyze_drums  -> math_drums  -> lag_drums  -> null_drums_ctrl
audiofilein_bass   -> analyze_bass   -> math_bass   -> lag_bass   -> null_bass_ctrl
audiofilein_guitar -> analyze_guitar -> math_guitar -> lag_guitar -> null_guitar_ctrl
audiofilein_piano  -> analyze_piano  -> math_piano  -> lag_piano  -> null_piano_ctrl
audiofilein_other  -> analyze_other  -> math_other  -> lag_other  -> null_other_ctrl
```

Recommended first mapping for the flower design:

| Stem | Visual control idea |
|------|---------------------|
| vocals | flower center glow, petal opening, emotional pulse |
| drums | particle bursts, sharp bloom hits, quick camera/light pulses |
| bass | whole flower scale, low-frequency breathing, core expansion |
| guitar | particle swirl, noise turbulence, orbit speed |
| piano | petal shimmer, sparkle intensity, fine twist amount |
| other | background atmosphere, slow field motion, residual texture |

Example parameter expressions:

```python
# flower scale / bass pulse
1 + op('null_bass_ctrl')[0] * 0.35

# particle birth / drum hits
20 + op('null_drums_ctrl')[0] * 120

# petal twist / piano shimmer
op('current_mood')['arousal'][0] * 0.8 + op('null_piano_ctrl')[0] * 1.2

# particle turbulence / guitar motion
0.2 + op('null_guitar_ctrl')[0] * 1.5
```

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
- `[MoodWave] Loading file: C:/.../uploads/run_<id>/current.wav`
- `[MoodWave] Loading stem drums: C:/.../uploads/run_<id>/stems/drums.wav`
- `[MoodWave] Loading stem bass: C:/.../uploads/run_<id>/stems/bass.wav`
- `[MoodWave] Transport: play`

Once the timeline payload has been received, the `current_mood` Script CHOP
will start tracking the audio's playhead with no extra plumbing.
