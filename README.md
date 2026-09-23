# Face Locking

Manual-selection identity locking, built on top of a face recognition and
tracking pipeline. You choose which enrolled person to lock onto; the
system recognizes them, locks on, ignores everyone else, keeps tracking
them through brief recognition hiccups, steers a servo to follow them, and
logs every detected action (movement direction, blinks, smiles, frowns)
to a timestamped history file — with the servo physically controlled by
an ESP8266 over MQTT.

This README is ordered by what matters most: get **`src/face_lock.py`**
running first — that's the primary deliverable. Everything else (basic
recognition-only tools, ESP8266/hardware, MQTT broker choice, other
tracking scripts) is supporting material below it.

---

## 1. Prerequisites

- Python 3.10–3.13
- A webcam
- Windows, macOS, or Linux (examples below use PowerShell; adjust for your shell)
- *(For hardware control)* an ESP8266 board (NodeMCU/Wemos), a hobby
  servo, a 5V power supply for the servo, and a 2.4GHz WiFi network

---

## 2. Install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
pip install python-dotenv pyserial adafruit-ampy
```

`opencv-contrib-python` is required, not just `opencv-python` — action
detection uses `cv2.face`, which only ships in the `contrib` build. **Only
one of `opencv-python` / `opencv-contrib-python` can be installed at a
time** (both provide the `cv2` module and will conflict):

```powershell
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-contrib-python
```

Verify:
```powershell
python -c "import cv2; print(cv2.__version__); print('has cv2.face:', hasattr(cv2, 'face'))"
```
`has cv2.face` must print `True`.

---

## 3. Download the model files

Three pretrained models go under `models/` (not committed — see
`.gitignore` — download once):

| File | Purpose | Source |
|---|---|---|
| `models/face_detection_yunet_2023mar.onnx` | Face detection + 5-point landmarks | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| `models/embedder_arcface.onnx` | 512-D face embedding for recognition | InsightFace `buffalo_l` pack (`w600k_r50.onnx`) |
| `models/lbfmodel.yaml` | 68-point landmarks for action detection | [kurnianggoro/GSOC2017](https://github.com/kurnianggoro/GSOC2017) |

```powershell
mkdir models -ErrorAction SilentlyContinue

# ArcFace embedder (via insightface, one-time download trigger)
pip install insightface
python -c "from insightface.app import FaceAnalysis; a=FaceAnalysis(name='buffalo_l'); a.prepare(ctx_id=0)"
copy "$env:USERPROFILE\.insightface\models\buffalo_l\w600k_r50.onnx" "models\embedder_arcface.onnx"

# 68-point landmark model (for action detection)
curl.exe -L -o models\lbfmodel.yaml https://raw.githubusercontent.com/kurnianggoro/GSOC2017/master/data/lbfmodel.yaml
```

For `face_detection_yunet_2023mar.onnx`, download manually from the
OpenCV Zoo link above and place it in `models/`.

Confirm all three are present: `dir models`

**This runs CPU-only** — `onnxruntime` defaults to `CPUExecutionProvider`
(see `src/embed.py`); no GPU is required or used anywhere in this project.

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

Prefix the MQTT topics with something unique to you if using a public
broker (see [§8](#8-mqtt-broker)) to avoid colliding with other users'
traffic on the same shared broker.

---

## 5. Enroll the identities you want to lock onto

```powershell
python -m src.enroll
```
Enter a name, then:
- `a` — toggle auto-capture (aim for 15+ varied-angle samples)
- `SPACE` — capture one frame manually
- `s` — **save** to the database (required — capturing alone does not
  save; press `s` before quitting)
- `q` — quit

Repeat for each person. `src/face_lock.py` (next section) can only lock
onto identities that exist in this database.

---

## 6. Face Locking — `src/face_lock.py` (primary deliverable)

```powershell
python -m src.face_lock
```

### What it does

1. **Manual selection** — on startup, it lists every enrolled identity and
   asks you to pick one by number or name. This is the identity that will
   be locked onto — the system does not auto-select whoever it happens to
   recognize first.
2. **Lock activation** — the lock engages once the *selected* identity is
   confidently recognized for `TARGET_CONFIRMATION_FRAMES` consecutive
   recognition cycles (default 3).
3. **Lock persistence** — once locked, every other detected face is
   ignored entirely (not drawn, not tracked, not logged). The lock
   survives brief recognition misses and is only released after
   `TARGET_LOST_FRAMES` consecutive frames (default 15) with no sign of
   the target — not after a single missed frame.
4. **Mandatory action detection (locked identity only)** — while locked,
   the system detects:
   - **Movement direction** — left / right / center relative to frame
     center, logged only when the direction changes (not every frame)
   - **Blinks** — via Eye Aspect Ratio on 68-point landmarks
   - **Smiles** / **Frowns** — via mouth width and corner elevation
5. **Action history log** — every action is appended as one JSON object
   per line to `data/action_log/<identity_name>.jsonl`:
   ```json
   {"timestamp": "2026-09-23T14:02:11+00:00", "action": "lock_acquired", "description": "Locked onto 'prince'."}
   {"timestamp": "2026-09-23T14:02:14+00:00", "action": "movement", "description": "'prince' moved right of center."}
   {"timestamp": "2026-09-23T14:02:16+00:00", "action": "blink", "description": "'prince' blinked."}
   {"timestamp": "2026-09-23T14:02:19+00:00", "action": "smile", "description": "'prince' started smiling."}
   ```
   Every entry has exactly `timestamp`, `action`, and `description`.

### Keys

`q` quit &nbsp;·&nbsp; `c` recenter servo &nbsp;·&nbsp; `d` toggle debug HUD

### Tuning

Action-detection thresholds (`SMILE_WIDTH_RATIO`, `SMILE_CORNER_RISE_PX`,
`FROWN_CORNER_DROP_PX`, `EAR_BLINK_THRESHOLD`) live at the top of
`src/expressions.py` — heuristic, not a trained classifier by design
(explainable, CPU-only logic). Tune for your camera/lighting if detection
feels too sensitive or not sensitive enough.

You can verify the whole software side (selection → lock → action
logging) using just the console output and the `cv2.imshow` window — the
ESP8266/servo is optional for confirming this part works; MQTT publishes
will simply go out even without anything subscribed on the other end.

---

## 7. ESP8266 + servo setup (physical tracking output)

### 7.1 Wiring

- Servo signal wire → **D5 (GPIO14)** on the ESP8266
- Servo power (red) → a proper **5V supply** — not the ESP8266's 3.3V pin;
  servos draw 200mA–1A under load, more than the board's regulator or a
  laptop USB port can reliably provide alongside WiFi
- Servo ground (black/brown) → **shared ground** with the ESP8266 — the
  most common wiring mistake, and it causes "receives signal but doesn't
  move" symptoms

### 7.2 WiFi must be 2.4GHz

The ESP8266's radio **cannot join 5GHz-only networks**. Check what's
visible from your PC:
```powershell
netsh wlan show networks mode=bssid
```
The `Band` field must show `2.4 GHz`. If your router is 5GHz-only, enable
a separate 2.4GHz SSID in its admin panel, or use a phone hotspot set to
2.4GHz for testing.

### 7.3 Configure `esp8266/main.py`

```python
WIFI_SSID = "your-2.4ghz-ssid"
WIFI_PASSWORD = "your-password"

MQTT_BROKER = "broker.emqx.io"     # match .env's MQTT_BROKER
MQTT_PORT = 1883

SERVO_TOPIC = "yourname/servo/angle"       # match .env's MQTT_SERVO_TOPIC
STATUS_TOPIC = "yourname/tracking/status"  # match .env's MQTT_STATUS_TOPIC
```

Both the ESP8266 and your PC must be able to reach the same MQTT broker
(see [§8](#8-mqtt-broker)).

### 7.4 Flash with `ampy`

Close any serial monitor connections first (ampy needs exclusive port access).

```powershell
cd esp8266
ampy --port COM5 put umqtt_simple.py
ampy --port COM5 put main.py
```
(replace `COM5` with your board's actual port)

Confirm: `ampy --port COM5 ls`

Reset the board and watch the log:
```powershell
python src\esp_logger.py
```
(update `PORT` inside `esp_logger.py` first)

Expected order: `[SERVO] angle: 90` → `[WiFi] Connected` →
`[MQTT] Connected, result: 0` → `[MQTT] Subscribed: ...` → `[SYSTEM] Ready.`

### 7.5 Run it end-to-end

With the ESP8266 powered, connected, and subscribed, run
`python -m src.face_lock` on your PC — the servo should now physically
follow the locked identity as they move.

---

## 8. MQTT broker

Pick one — both `.env` and `esp8266/main.py` must point at the same
broker and topics.

**Public broker (fastest to get running):**
```
broker.emqx.io, port 1883
```
Prefix topics with something unique to you to avoid collisions on the
shared broker. Unauthenticated — anyone who guesses your topic name could
publish to it.

**Local broker (private, recommended once things work):**
Install [Mosquitto](https://mosquitto.org/download/), configure it to
listen on your LAN IP with anonymous access, point both devices at your
PC's local network IP (`ipconfig`). Both devices must be on the *same*
WiFi network/subnet.

---

## 9. Supporting scripts

These were the intermediate build stages toward `face_lock.py`. Not
required for the Face Locking deliverable, but useful for isolating
problems (e.g. confirming recognition alone works before adding lock/log
logic on top).

| Script | Purpose |
|---|---|
| `python -m src.evaluate` | Suggests a `RECOGNITION_THRESHOLD` from your enrolled crops |
| `python -m src.recognize` | Recognition only, no tracking, no hardware — `q` quit, `r` reload DB, `+`/`-` threshold |
| `python -m src.track` | Basic tracking, no identity — follows whoever's biggest/most-centered |
| `python -m src.track_with_recognition` | Identity-locked tracking, auto-locks onto first confirmed identity (no manual selection, no action log — the precursor to `face_lock.py`) |
| `python -m src.track_with_expressions` | Same as above, plus live expression HUD (no manual selection, no action log) |

---

## 10. Project structure

```
├── requirements.txt
├── .env                          # not committed — see §4
├── models/                       # not committed — see §3
│   ├── face_detection_yunet_2023mar.onnx
│   ├── embedder_arcface.onnx
│   └── lbfmodel.yaml
├── data/                         # not committed — personal biometric data
│   ├── db/face_db.npz            # enrolled identity embeddings
│   ├── enroll/<name>/*.jpg       # raw aligned enrollment crops
│   └── action_log/<name>.jsonl   # face_lock.py action history
├── esp8266/
│   ├── main.py                   # WiFi + MQTT + servo control firmware
│   └── umqtt_simple.py           # minimal MQTT client for MicroPython
└── src/
    ├── config.py                 # loads .env into a typed Config object
    ├── detect_landmarks.py       # YuNet face detection + 5-point landmarks
    ├── align.py                  # 5-point affine alignment to 112x112
    ├── embed.py                  # ArcFace ONNX embedding extraction
    ├── enroll.py                 # enrollment tool (§5)
    ├── evaluate.py                # threshold tuning tool (§9)
    ├── recognize.py               # Recognizer class + standalone demo (§9)
    ├── expressions.py             # blink/smile/frown via cv2.face LBF landmarks
    ├── face_lock.py               # PRIMARY: manual selection, lock, action log (§6)
    ├── track.py                   # basic tracking, no identity (§9)
    ├── track_with_recognition.py  # auto-lock tracking, precursor to face_lock.py (§9)
    ├── track_with_expressions.py  # auto-lock tracking + expression HUD (§9)
    ├── mqtt_bridge.py             # PC-side MQTT publish wrapper
    └── esp_logger.py              # live serial log viewer for the ESP8266
```

---

## 11. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No enrolled identities found` at startup | Nothing enrolled yet, or enrolled but never saved | Run `python -m src.enroll`, remember to press `s` before quitting |
| `has cv2.face: False` | Plain `opencv-python` installed, not `opencv-contrib-python` | See §2 |
| `LBF model not found` | `models/lbfmodel.yaml` missing | See §3 |
| `[WiFi] waiting...` forever, then `RuntimeError` | ESP8266 is 2.4GHz-only, your SSID is 5GHz | See §7.2 |
| `MQTTException: No MQTT CONNACK` | Malformed CONNECT packet (fixed in the `umqtt_simple.py` in this repo), wrong broker/port, or broker rejecting anonymous connections | Confirm you have the current `umqtt_simple.py`; test the broker with `mosquitto_pub`/`sub` first |
| `OSError: [Errno 110] ETIMEDOUT` on MQTT connect | ESP8266 and the MQTT broker are on different, unreachable networks | Confirm both devices' `ipconfig`/`ifconfig` show the same subnet, or use a public broker |
| Servo receives MQTT angles but doesn't move | Most often power/ground wiring — servo needs a real 5V rail and a ground shared with the ESP8266; also try disabling WiFi temporarily to rule out ESP8266 software-PWM/WiFi timing jitter | See §7.1; try manually rotating the horn by hand to check for resistance |
| `TypeError: Object of type bool is not JSON serializable` on MQTT publish | numpy `bool_` (from OpenCV/numpy comparisons) instead of native Python `bool` | Fixed in the current `expressions.py` (`bool(...)` wraps all comparison results) |
| Lock never activates even though the right person is on camera | Selected name doesn't exactly match an enrolled name, or lighting/angle is hurting recognition confidence | Re-check `rec.names` matches what you typed; try `python -m src.recognize` first to sanity-check recognition alone |