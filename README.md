# Face Recognition, Tracking & Expression System

A pipeline that detects and recognizes faces from a webcam, locks onto and
tracks one identified person by steering a servo, and reads their facial
expressions (blinks, smiles, frowns) — with the servo physically controlled
by an ESP8266 over MQTT.

This README is ordered by what to get running first: the **software
pipeline on your PC** (detection → enrollment → recognition → tracking →
expressions) works and can be fully tested with just a webcam, before you
touch any hardware. The **ESP8266 + servo** section comes after, since it
only matters once the software side is already producing angles to send it.

---

## 1. Prerequisites

- Python 3.10–3.13
- A webcam
- Windows, macOS, or Linux (examples below use PowerShell; adjust for your shell)
- *(For hardware control later)* an ESP8266 board (NodeMCU/Wemos), a hobby
  servo, a 5V power supply for the servo, and a 2.4GHz WiFi network

---

## 2. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install python-dotenv pyserial adafruit-ampy
```

`opencv-contrib-python` (already in `requirements.txt`) is required, not
just `opencv-python` — the expression module uses `cv2.face`, which only
ships in the `contrib` build. **Only one of `opencv-python` /
`opencv-contrib-python` can be installed at a time** (they both provide
the `cv2` module and will conflict):

```powershell
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python
```

Verify it worked:
```powershell
python -c "import cv2; print(cv2.__version__); print('has cv2.face:', hasattr(cv2, 'face'))"
```
`has cv2.face` must print `True`.

---

## 3. Download the model files

Three pretrained models are needed under `models/` (not committed to the
repo — see `.gitignore` — download them once):

| File | Purpose | Source |
|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | Face detection + 5-point landmarks | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `models/embedder_arcface.onnx` | 512-D face embedding for recognition | InsightFace `buffalo_l` pack (`w600k_r50.onnx`) |
| `models/lbfmodel.yaml` | 68-point landmarks for blink/smile/frown | [kurnianggoro/GSOC2017](https://github.com/kurnianggoro/GSOC2017) |

```powershell
mkdir models -ErrorAction SilentlyContinue

# ArcFace embedder (via insightface, one-time download trigger)
pip install insightface
python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l'); a.prepare(ctx_id=0)"
copy "$env:USERPROFILE\.insightface\models\buffalo_l\w600k_r50.onnx" "models\embedder_arcface.onnx"

# 68-point landmark model (for expressions)
curl.exe -L -o models\lbfmodel.yaml https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
```

For `face_detection_yunet_2023mar.onnx`, download it manually from the
OpenCV Zoo link above and place it in `models/`.

Confirm all three are present:
```powershell
dir models
```

---

## 4. Configure `.env`

Create a `.env` file in the project root:

```env
CAMERA_INDEX=0
MIRROR_CAMERA=true

ARC_FACE_MODEL=models/embedder_arcface.onnx
RECOGNITION_THRESHOLD=0.34

MQTT_BROKER=broker.emqx.io
MQTT_PORT=1883
MQTT_SERVO_TOPIC=yourname/servo/angle
MQTT_STATUS_TOPIC=yourname/tracking/status
MQTT_DATA_TOPIC=yourname/tracking/data

SERVO_MIN_ANGLE=10
SERVO_MAX_ANGLE=170
SERVO_CENTER_ANGLE=90
SERVO_MIN_CHANGE=1

TRACKING_DEAD_ZONE=35
TRACKING_KP=0.08
TRACKING_MAX_STEP=4

RECOGNITION_EVERY_N_FRAMES=5
TARGET_CONFIRMATION_FRAMES=3
TARGET_LOST_FRAMES=15
```

Prefix `MQTT_SERVO_TOPIC` / `MQTT_STATUS_TOPIC` / `MQTT_DATA_TOPIC` with
something unique to you (e.g. your name) if using a public broker like
`broker.emqx.io` — this avoids colliding with other people's traffic on
the same shared broker. See [§7](#7-mqtt-broker) for broker options.

---

## 5. Software pipeline — run in this order

### 5.1 Enroll faces

```powershell
python -m src.enroll
```
Enter a name, then:
- `a` — toggle auto-capture (recommended: aim for 15+ varied-angle samples)
- `SPACE` — capture one frame manually
- `s` — **save** the identity to the database (required — capturing alone
  does not save; the crops are cached, but `s` is what writes
  `data/db/face_db.npz`)
- `q` — quit

Repeat for each person you want recognized.

### 5.2 (Optional) Evaluate the recognition threshold

```powershell
python -m src.evaluate
```
Uses your enrolled crops to suggest a `RECOGNITION_THRESHOLD` value that
balances false accepts vs. false rejects. Put the suggested value in `.env`.

### 5.3 Test recognition alone (no tracking, no hardware)

```powershell
python -m src.recognize
```
Shows live bounding boxes with names or "Unknown", and the match distance.
Keys: `q` quit, `r` reload DB, `+`/`-` adjust threshold live.

### 5.4 Basic face tracking (no identity — tracks whoever is biggest/most-centered)

```powershell
python -m src.track
```
Publishes servo angles over MQTT as it tracks the most prominent face.
Doesn't require an enrolled identity.

### 5.5 Identity-locked tracking (recommended)

```powershell
python -m src.track_with_recognition
```
Detects every frame, re-identifies every `RECOGNITION_EVERY_N_FRAMES`
frames, and only locks onto (tracks) a face once the same identity is
confirmed for `TARGET_CONFIRMATION_FRAMES` consecutive checks. Holds the
lock for up to `TARGET_LOST_FRAMES` frames if the person briefly leaves
frame. Keys: `q` quit, `c` recenter, `r` reload DB, `d` toggle debug HUD.

### 5.6 Identity-locked tracking **with expressions** (full pipeline)

```powershell
python -m src.track_with_expressions
```
Everything in 5.5, plus once a target is locked: blink detection (via Eye
Aspect Ratio on 68-point landmarks), and smile/frown detection (via mouth
width and corner elevation). The HUD shows live EAR value, blink count,
and current expression (`Smiling` / `Frowning` / `Blinking` / `Neutral`).
Expression results are also published over MQTT on `MQTT_DATA_TOPIC`
alongside tracking data. Same keys as 5.5.

> Expression thresholds (`SMILE_WIDTH_RATIO`, `SMILE_CORNER_RISE_PX`,
> `FROWN_CORNER_DROP_PX`, `EAR_BLINK_THRESHOLD`) are heuristics defined at
> the top of `src/expressions.py` — tune them for your camera/lighting if
> detection feels too sensitive or not sensitive enough.

At this point the whole software pipeline is verifiable end-to-end just
by watching the console log show MQTT angle publishes — you don't need
the ESP8266 connected yet to confirm detection/recognition/tracking/
expressions are all working correctly.

---

## 6. ESP8266 + servo setup

### 6.1 Wiring

- Servo signal wire → **D5 (GPIO14)** on the ESP8266
- Servo power (red) → a proper **5V supply** — not the ESP8266's 3.3V pin;
  servos draw 200mA–1A under load, more than the board's regulator or a
  laptop USB port can reliably provide alongside WiFi
- Servo ground (black/brown) → **shared ground** with the ESP8266 — this is
  the single most common wiring mistake and causes "receives signal but
  doesn't move" symptoms

### 6.2 Important: WiFi must be 2.4GHz

The ESP8266's radio **cannot join 5GHz-only networks**. If your router
only broadcasts one 5GHz SSID, either:
- enable a separate 2.4GHz SSID in the router admin panel, or
- use a phone hotspot set to 2.4GHz for testing.

Check what's visible from your PC:
```powershell
netsh wlan show networks mode=bssid
```
Look at the `Band` field — it must show `2.4 GHz` for the ESP8266 to join.

### 6.3 Configure `esp8266/main.py`

```python
WIFI_SSID = "your-2.4ghz-ssid"
WIFI_PASSWORD = "your-password"

MQTT_BROKER = "broker.emqx.io"     # match .env's MQTT_BROKER
MQTT_PORT = 1883

SERVO_TOPIC = "yourname/servo/angle"       # match .env's MQTT_SERVO_TOPIC
STATUS_TOPIC = "yourname/tracking/status"  # match .env's MQTT_STATUS_TOPIC
```

**Both the ESP8266 and your PC must be able to reach the same MQTT
broker.** If using a local broker instead of a public one, both devices
need to be on the *same LAN* (see [§7](#7-mqtt-broker)).

### 6.4 Flash the ESP8266 with `ampy`

Close any serial monitor connections first (ampy needs exclusive port access).

```powershell
cd esp8266
ampy --port COM5 put umqtt_simple.py
ampy --port COM5 put main.py
```
(replace `COM5` with your board's actual port — check Device Manager)

Confirm what's on the board:
```powershell
ampy --port COM5 ls
```

Reset the board (power cycle, or press its reset button) and watch the log:
```powershell
python src\esp_logger.py
```
(update `PORT` inside `esp_logger.py` to match your COM port first)

You should see, in order: `[SERVO] angle: 90`, `[WiFi] Connected`,
`[MQTT] Connected, result: 0`, `[MQTT] Subscribed: ...`, `[SYSTEM] Ready.`

### 6.5 Run the full system

With the ESP8266 powered on, connected, and subscribed, run any of the
tracking scripts from [§5](#5-software-pipeline--run-in-this-order) (5.4,
5.5, or 5.6) on your PC. The servo should now physically move as it tracks.

---

## 7. MQTT broker

Two options — pick one and make sure **both** `.env` and
`esp8266/main.py` point at the same broker and topics.

**Public broker (easiest to get running, no setup):**
```
broker.emqx.io, port 1883
```
Use topic names prefixed with something unique to you to avoid collisions
with other users on the shared broker (see `.env` example in §4). Note
this is unauthenticated and public — anyone who guesses your topic name
could publish to it.

**Local broker (private, recommended once things are working):**
Install [Mosquitto](https://mosquitto.org/download/), configure it to
listen on your LAN IP with anonymous access, and point both devices at
your PC's local network IP (find it with `ipconfig`). Both the ESP8266
and your PC must be joined to the *same* WiFi network/subnet for this to
work — check with `ipconfig` (PC) that the IP range matches the ESP8266's
`[WiFi] Connected` log output.

---

## 8. Project structure

```
├── requirements.txt
├── .env                          # not committed — see §4
├── models/                       # not committed — see §3
│   ├── face_detection_yunet_2023mar.onnx
│   ├── embedder_arcface.onnx
│   └── lbfmodel.yaml
├── data/                         # not committed — personal biometric data
│   ├── db/face_db.npz            # enrolled identity embeddings
│   └── enroll/<name>/*.jpg       # raw aligned enrollment crops
├── esp8266/
│   ├── main.py                   # WiFi + MQTT + servo control firmware
│   └── umqtt_simple.py           # minimal MQTT client for MicroPython
└── src/
    ├── config.py                 # loads .env into a typed Config object
    ├── detect_landmarks.py       # YuNet face detection + 5-point landmarks
    ├── align.py                  # 5-point affine alignment to 112x112
    ├── embed.py                  # ArcFace ONNX embedding extraction
    ├── enroll.py                 # enrollment tool (§5.1)
    ├── evaluate.py                # threshold tuning tool (§5.2)
    ├── recognize.py               # Recognizer class + standalone demo (§5.3)
    ├── track.py                   # basic tracking, no identity (§5.4)
    ├── track_with_recognition.py  # identity-locked tracking (§5.5)
    ├── expressions.py             # blink/smile/frown via cv2.face LBF landmarks
    ├── track_with_expressions.py  # full pipeline: tracking + expressions (§5.6)
    ├── mqtt_bridge.py             # PC-side MQTT publish wrapper
    └── esp_logger.py              # live serial log viewer for the ESP8266
```

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `has cv2.face: False` | Plain `opencv-python` installed, not `opencv-contrib-python` | See §2 |
| `LBF model not found` | `models/lbfmodel.yaml` missing | See §3 |
| `[WiFi] waiting...` forever, then `RuntimeError` | ESP8266 is 2.4GHz-only, your SSID is 5GHz | See §6.2 |
| `MQTTException: No MQTT CONNACK` | Malformed CONNECT packet (fixed in the `umqtt_simple.py` in this repo), wrong broker/port, or broker rejecting anonymous connections | Confirm you have the current `umqtt_simple.py`; test the broker with `mosquitto_pub`/`sub` first |
| `OSError: [Errno 110] ETIMEDOUT` on MQTT connect | ESP8266 and the MQTT broker are on different, unreachable networks | Confirm both devices' `ipconfig`/`ifconfig` show the same subnet, or use a public broker |
| Servo receives MQTT angles but doesn't move | Most often power/ground wiring — servo needs a real 5V rail and a ground shared with the ESP8266; also try disabling WiFi temporarily to rule out ESP8266 software-PWM/WiFi timing jitter | See §6.1; try manually rotating the horn by hand to check for resistance |
| `TypeError: Object of type bool is not JSON serializable` on MQTT publish | numpy `bool_` (from OpenCV/numpy comparisons) instead of native Python `bool` | Fixed in the current `expressions.py` (`bool(...)` wraps around all comparison results) |
| `0 identities` when running `recognize.py` after enrolling | Forgot to press `s` during enrollment — capturing crops alone does not save the database | Rerun `enroll.py`, it reloads existing crops automatically, then press `s` |