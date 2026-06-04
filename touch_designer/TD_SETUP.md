# MoodWave TouchDesigner Setup

This document describes the final TouchDesigner side of MoodWave. The backend analyzes the uploaded song, writes the main WAV and Demucs stem WAV files, the frontend sends one complete timeline payload to the OSC bridge, and TouchDesigner uses that data to drive the audio-reactive visual.

## Runtime Pipeline

```text
Frontend upload
    -> Backend analysis
        -> main audio WAV + six Demucs stems + MuQ-BiGRU timeline
    -> Frontend WebSocket message
        -> touch_designer/osc_bridge.py
    -> TouchDesigner OSC inputs
        -> audio playback, timeline lookup, stem controls, visuals
    -> td_mjpeg_server.py
        -> browser preview at http://localhost:9000/video
```

TouchDesigner owns playback. The frontend only sends timeline data, play, pause, and seek commands.

## Data Sent To TouchDesigner

The OSC bridge receives browser WebSocket messages on `ws://localhost:7011` and forwards OSC to TouchDesigner on UDP port `7000`.

| Address | Type | Purpose |
| --- | --- | --- |
| `/moodwave/filepath` | string | Absolute path to the main `current.wav` |
| `/moodwave/duration` | float | Song duration in seconds |
| `/moodwave/chunk_duration` | float | Timeline step size, normally `0.5` seconds |
| `/moodwave/num_chunks` | int | Number of timeline points |
| `/moodwave/stems/vocals` | string | Absolute path to `vocals.wav` |
| `/moodwave/stems/drums` | string | Absolute path to `drums.wav` |
| `/moodwave/stems/bass` | string | Absolute path to `bass.wav` |
| `/moodwave/stems/guitar` | string | Absolute path to `guitar.wav` |
| `/moodwave/stems/piano` | string | Absolute path to `piano.wav` |
| `/moodwave/stems/other` | string | Absolute path to `other.wav` |
| `/moodwave/timeline/valence` | float array | Valence per timeline point |
| `/moodwave/timeline/arousal` | float array | Arousal per timeline point |
| `/moodwave/timeline/tempo` | float array | Tempo per timeline point |
| `/moodwave/timeline/energy` | float array | Energy per timeline point |
| `/moodwave/timeline/brightness` | float array | Spectral brightness per timeline point |
| `/moodwave/timeline/dominant` | int array | Dominant mood index per timeline point |
| `/moodwave/timeline/mood/{name}` | float array | Probability for one mood per timeline point |
| `/moodwave/transport` | `0` or `1` | Pause or play |
| `/moodwave/seek` | float | Seek time in seconds |

Mood names are `energetic`, `happy`, `calm`, `romantic`, `sad`, and `angry`.

Demucs outputs six synchronized 44.1 kHz stereo PCM WAV stems: `vocals`, `drums`, `bass`, `guitar`, `piano`, and `other`.

## Required Operators

Create these operators in `/project1` unless your network uses another root.

| Operator | Name | Role |
| --- | --- | --- |
| OSC In DAT | `oscin_ctrl` | Receives file paths, stems, transport, seek, and metadata |
| Text DAT | `osc_callbacks` | Callback script for `oscin_ctrl` |
| OSC In CHOP | `oscin_timeline` | Receives timeline arrays |
| Audio File In CHOP | `audiofilein1` | Main mixed song playback |
| Audio File In CHOP | `audiofilein_vocals` | Vocals stem playback |
| Audio File In CHOP | `audiofilein_drums` | Drums stem playback |
| Audio File In CHOP | `audiofilein_bass` | Bass stem playback |
| Audio File In CHOP | `audiofilein_guitar` | Guitar stem playback |
| Audio File In CHOP | `audiofilein_piano` | Piano stem playback |
| Audio File In CHOP | `audiofilein_other` | Other stem playback |
| Timer CHOP | `playhead_timer` | Current playback time for timeline lookup |
| Script CHOP | `current_mood` | Samples the timeline at the current playhead time |

Set the OSC In DAT `oscin_ctrl`:

- `Network Port`: `7000`
- `Active`: on
- `Callbacks DAT`: `osc_callbacks`

Set the OSC In CHOP `oscin_timeline`:

- `Network Port`: `7000`
- `Address Scope`: `/moodwave/timeline/*`

Set every Audio File In CHOP:

- `File`: blank
- `Play`: off
- `Loop`: off

Set the Timer CHOP `playhead_timer`:

- `Play`: off
- Start at `0`
- The callback controls reset, seek, play, and duration from OSC

## OSC Callback Script

Paste this into the Text DAT named `osc_callbacks`.

```python
STEMS = ('vocals', 'drums', 'bass', 'guitar', 'piano', 'other')
STEM_AUDIO_OPS = ['audiofilein_' + name for name in STEMS]


def _all_audio_ops():
    names = ['audiofilein1'] + STEM_AUDIO_OPS
    result = []
    for name in names:
        audio = op(name)
        if audio is not None:
            result.append(audio)
    return result


def _load_audio(audio, filepath):
    audio.par.file.val = filepath
    if hasattr(audio.par, 'loop'):
        audio.par.loop = 0
    audio.par.cuepoint.val = 0
    audio.par.cuepulse.pulse()


def _reset_timer(timer):
    if timer is None:
        return
    if hasattr(timer.par, 'initialize'):
        timer.par.initialize.pulse()
    if hasattr(timer.par, 'start'):
        timer.par.start.pulse()
    if hasattr(timer.par, 'play'):
        timer.par.play = 0


def _seek_timer(timer, seconds):
    if timer is None:
        return
    if hasattr(timer.par, 'cuepoint'):
        timer.par.cuepoint.val = seconds
    if hasattr(timer.par, 'cuepulse'):
        timer.par.cuepulse.pulse()


def _set_timer_duration(timer, seconds):
    if timer is None:
        return
    for par_name in ('length', 'duration'):
        if hasattr(timer.par, par_name):
            getattr(timer.par, par_name).val = seconds
            return


def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    timer = op('playhead_timer')

    if address == '/moodwave/filepath' and len(args) > 0:
        filepath = args[0]
        audio = op('audiofilein1')
        if audio:
            print('[MoodWave] Loading file:', filepath)
            _load_audio(audio, filepath)
        _reset_timer(timer)

    elif address == '/moodwave/duration' and len(args) > 0:
        duration = float(args[0])
        dat.parent().store('moodwave_duration', duration)
        _set_timer_duration(timer, duration)

    elif address == '/moodwave/chunk_duration' and len(args) > 0:
        dat.parent().store('moodwave_chunk_duration', float(args[0]))

    elif address == '/moodwave/num_chunks' and len(args) > 0:
        dat.parent().store('moodwave_num_chunks', int(args[0]))

    elif address.startswith('/moodwave/stems/') and len(args) > 0:
        stem_name = address.split('/')[-1]
        stem = op('audiofilein_' + stem_name)
        if stem:
            print('[MoodWave] Loading stem {}: {}'.format(stem_name, args[0]))
            _load_audio(stem, args[0])

    elif address == '/moodwave/seek' and len(args) > 0:
        seconds = float(args[0])
        for audio in _all_audio_ops():
            audio.par.cuepoint.val = seconds
            audio.par.cuepulse.pulse()
        _seek_timer(timer, seconds)
        print('[MoodWave] Seek -> {:.2f}s'.format(seconds))

    elif address == '/moodwave/transport' and len(args) > 0:
        play = int(args[0])
        for audio in _all_audio_ops():
            audio.par.play = play
        if timer is not None and hasattr(timer.par, 'play'):
            timer.par.play = play
        print('[MoodWave] Transport:', 'play' if play else 'pause')
```

## Current Mood Script CHOP

Create a Script CHOP named `current_mood`. It samples the timeline arrays using the current timer position and outputs one-sample control channels for the active moment in the song.

```python
import colorsys

MOODS = ('energetic', 'happy', 'calm', 'romantic', 'sad', 'angry')
FEATURES = ('tempo', 'energy', 'brightness')


def _chan(chop, name):
    if chop is None:
        return None
    try:
        return chop[name]
    except Exception:
        return None


def _sample(ch, idx, default=0.0):
    if ch is None or ch.numSamples == 0:
        return default
    idx = max(0, min(idx, ch.numSamples - 1))
    return float(ch[idx])


def _append(scriptOp, name, value):
    c = scriptOp.appendChan(name)
    c[0] = float(value)


def _current_seconds():
    timer = op('playhead_timer')
    for name in ('timer_seconds', 'seconds', 'time', 't'):
        ch = _chan(timer, name)
        if ch is not None:
            return float(ch[0])

    audio = op('audiofilein1')
    for name in ('t', 'time', 'seconds'):
        ch = _chan(audio, name)
        if ch is not None:
            return float(ch[0])

    return 0.0


def onCook(scriptOp):
    scriptOp.clear()
    scriptOp.numSamples = 1

    timeline = op('oscin_timeline')
    chunk_dur = float(parent().fetch('moodwave_chunk_duration', 0.5))
    chunk_dur = max(chunk_dur, 0.001)
    chunk_idx = int(_current_seconds() // chunk_dur)

    val_ch = _chan(timeline, 'moodwave/timeline/valence')
    aro_ch = _chan(timeline, 'moodwave/timeline/arousal')

    valence = _sample(val_ch, chunk_idx, 0.0)
    arousal = _sample(aro_ch, chunk_idx, 0.0)

    _append(scriptOp, 'valence', valence)
    _append(scriptOp, 'arousal', arousal)

    for feature in FEATURES:
        ch = _chan(timeline, 'moodwave/timeline/' + feature)
        _append(scriptOp, feature, _sample(ch, chunk_idx, 0.0))

    mood_values = []
    for mood in MOODS:
        ch = _chan(timeline, 'moodwave/timeline/mood/' + mood)
        value = _sample(ch, chunk_idx, 0.0)
        mood_values.append(value)
        _append(scriptOp, mood, value)

    if sum(mood_values) > 0:
        dominant = max(range(len(mood_values)), key=lambda i: mood_values[i])
        confidence = mood_values[dominant]
    else:
        dominant = int(_sample(_chan(timeline, 'moodwave/timeline/dominant'), chunk_idx, 0.0))
        confidence = 0.0

    _append(scriptOp, 'dominant', dominant)
    _append(scriptOp, 'confidence', confidence)

    # Compact valence/arousal color helper.
    # Use this for quick color links, or use the mood channels for richer ramp blends.
    u = max(0.0, min(1.0, (valence + 1.0) * 0.5))
    a = max(0.0, min(1.0, (arousal + 1.0) * 0.5))
    hue = (220.0 + u * (55.0 - 220.0)) / 360.0
    sat = 0.35 + a * 0.45
    bri = 0.45 + (abs(valence) + abs(arousal)) * 0.20
    r, g, b = colorsys.hsv_to_rgb(hue, sat, min(0.85, bri))

    _append(scriptOp, 'r', r)
    _append(scriptOp, 'g', g)
    _append(scriptOp, 'b', b)
```

Useful channels from `current_mood`:

- `valence`, `arousal`
- `tempo`, `energy`, `brightness`
- `energetic`, `happy`, `calm`, `romantic`, `sad`, `angry`
- `dominant`, `confidence`
- `r`, `g`, `b`

## Stem Control Branches

For each stem, create a small CHOP chain that turns the audio into stable control values:

```text
audiofilein_vocals -> analyze_vocals -> math_vocals -> lag_vocals -> null_vocals_ctrl
audiofilein_drums  -> analyze_drums  -> math_drums  -> lag_drums  -> null_drums_ctrl
audiofilein_bass   -> analyze_bass   -> math_bass   -> lag_bass   -> null_bass_ctrl
audiofilein_guitar -> analyze_guitar -> math_guitar -> lag_guitar -> null_guitar_ctrl
audiofilein_piano  -> analyze_piano  -> math_piano  -> lag_piano  -> null_piano_ctrl
audiofilein_other  -> analyze_other  -> math_other  -> lag_other  -> null_other_ctrl
```

Recommended visual mapping for the flower network:

| Stem | Best visual use |
| --- | --- |
| `vocals` | Flower center glow, petal opening, emotional pulse |
| `drums` | Particle bursts, sharp bloom hits, quick light flashes |
| `bass` | Whole flower scale, low-frequency breathing, core expansion |
| `guitar` | Particle swirl, noise turbulence, orbit speed |
| `piano` | Petal shimmer, sparkle amount, gentle twist |
| `other` | Background atmosphere, slow field motion, residual texture |

Example expressions:

```python
# bass-driven flower scale
1 + op('null_bass_ctrl')[0] * 0.35

# drum-driven particle bursts
20 + op('null_drums_ctrl')[0] * 120

# piano shimmer with arousal influence
op('current_mood')['arousal'][0] * 0.5 + op('null_piano_ctrl')[0] * 0.8

# guitar turbulence
0.2 + op('null_guitar_ctrl')[0] * 1.5
```

Use Lag CHOPs aggressively on continuous controls. Short lag works for drums; longer lag works for twist, color, bloom, camera, and whole-shape scale.

## Color Pipeline

The final color setup should use both valence/arousal and mood probabilities:

```text
current_mood
    -> optional Lag CHOP for smoother mood/color changes
    -> ramp color controls
    -> Ramp TOP
    -> Lookup TOP
    -> HSV Adjust / Level / Luma Blur
    -> output TOP
```

Practical usage:

- Use `valence` and `arousal` to choose the overall color region.
- Use mood probability channels to weight mood palettes, especially when two moods are mixed.
- Lag the mood/color control channels so dominant mood changes do not jump visually.
- Keep Ramp TOP edge keys darker and softer, then let the center keys carry the mood color.
- Use Level TOP brightness and bloom controls with restrained values so the render keeps detail.

## MJPEG Browser Preview

Use `touch_designer/td_mjpeg_server.py` inside TouchDesigner.

1. Create a Text DAT, for example `td_mjpeg_server`.
2. Paste the server script into it.
3. Set `TOP_PATH` in the script to the final output TOP, usually `/project1/out1`.
4. Create an Execute DAT.
5. Enable the `Start` callback and run:

```python
exec(op('td_mjpeg_server').text)
```

6. Enable the `Frame Start` callback and run:

```python
op('td_mjpeg_server').module.update_frame()
```

The frontend reads the TouchDesigner preview from:

```text
http://localhost:9000/video
```

You can check server health in the browser:

```text
http://localhost:9000/health
```

## Startup Order

Recommended launcher flow:

1. From the project root, run the MoodWave launcher:

```powershell
.\start_moodwave.ps1
```

You can also double-click:

```text
start_moodwave.bat
```

The launcher starts the bridge, starts the backend, and opens the frontend at:

```text
http://127.0.0.1:8000/
```

2. Open the TouchDesigner project file:

```text
C:\Users\dhana\SUSTech\Semester_6\CS330_Multimedia_Information_Processing\testing\touch_designer\Ghost_Flower.95.toe
```

3. Confirm the MJPEG server prints that it is running on port `9000`.

4. Upload a song or load a saved analysis from the library.

Manual equivalent:

1. Start the OSC bridge:

```powershell
cd touch_designer
python osc_bridge.py
```

2. Start the backend:

```powershell
cd backend
uvicorn main:app --reload --port 8000
```

3. Open `touch_designer\Ghost_Flower.95.toe`.

4. Confirm the MJPEG server prints that it is running on port `9000`.

5. Open the frontend:

```text
http://127.0.0.1:8000/
```

6. Upload a song or load a saved analysis from the library.

Expected TouchDesigner Textport messages:

```text
[MoodWave] Loading file: C:/.../current.wav
[MoodWave] Loading stem vocals: C:/.../vocals.wav
[MoodWave] Loading stem drums: C:/.../drums.wav
[MoodWave] Loading stem bass: C:/.../bass.wav
[MoodWave] Loading stem guitar: C:/.../guitar.wav
[MoodWave] Loading stem piano: C:/.../piano.wav
[MoodWave] Loading stem other: C:/.../other.wav
[MoodWave] Seek -> 0.00s
[MoodWave] Transport: play
```

When the timeline arrives, `oscin_timeline` should contain multi-sample channels and `current_mood` should output one-sample control values for the currently playing moment.
