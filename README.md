# Filch

A Raspberry Pi AI camera surveillance system built around the [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/) (Sony IMX500).
Object detection runs entirely on the camera's neural network accelerator — no cloud, no GPU, no separate inference server.
The Pi CPU is used only to save images and send notifications.

Named after [Argus Filch](https://harrypotter.fandom.com/wiki/Argus_Filch), the Hogwarts caretaker who keeps a watchful eye on everything.

---

## Features

- **On-device AI inference** — the IMX500 chip runs the neural network; the host CPU is nearly idle during detection
- **Daylight-only operation** — optionally sleeps from sunset to sunrise, computed from your configured coordinates; disable for indoor or IR-lit use
- **Configurable object filtering** — ignore noisy classes (e.g. birds, cars) and/or follow only specific ones (e.g. person)
- **Push notifications** via [ntfy.sh](https://ntfy.sh/) — free, no account required, works with the iOS/Android app
- **Annotated JPEG archive** — full-resolution images with bounding boxes saved per detection event, organised by date
- **Web preview** — downscaled JPEG written to `/var/www/html/` for a quick live view in a browser
- **Timelapse recording** — one frame every 10 minutes (configurable) regardless of detections

---

## Real-life usage examples

**Garden wildlife monitor** — mount the camera overlooking a garden or bird feeder. Set `follow = ["bird", "cat", "fox", "dog"]` to receive a phone notification with a photo whenever an animal enters the frame.

**Front door / driveway watch** — point the camera at your entrance. With `follow = ["person", "car"]` you get an ntfy notification the moment a visitor or vehicle appears, with a direct link to the saved image.

**Delivery detection** — keep the camera on a porch. Filter out everything except `person` and subscribe to the ntfy channel on your phone; you'll know the moment the courier arrives.

**Pest monitoring** — set `follow = ["cat"]` and `ignore = ["person", "car", "bird"]` to track a neighbourhood cat that visits your garden without being spammed by unrelated detections.

**Security audit log** — run without any follow/ignore filters to capture every detection above the confidence threshold. Images are archived under `database/YYMMDD/` for later review.

---

## Hardware requirements

| Component | Notes |
|-----------|-------|
| Raspberry Pi (any model with CSI connector) | Tested on RPi 2 (500 MB RAM) and the latest RPi 4. Should work smoothly on anything in between. |
| [Raspberry Pi AI Camera](https://www.raspberrypi.com/products/ai-camera/) | Uses the Sony IMX500 with integrated NPU |
| microSD card or USB storage | Images accumulate quickly at full resolution (2028×1520) |

---

## Dependencies

### System packages

```bash
sudo apt update
sudo apt install python3-picamera2 python3-opencv python3-numpy \
                   imx500-all python3-pip
```

`imx500-all` pulls in the IMX500 firmware and the default set of pre-trained models (installed under `/usr/share/imx500-models/`).

### Python packages

```bash
pip3 install suntime tzlocal requests
```

Python 3.11 or newer is required (uses the built-in `tomllib`). Raspberry Pi OS Bookworm ships with Python 3.11.

---

## Installation

```bash
git clone https://github.com/kbat/filch.git
cd filch
```

No build step is needed. The script runs directly.

### Optional: web preview

If you want the live preview feature, install a web server:

```bash
sudo apt install -y nginx
```

Filch writes `obj.jpg` and `timelapse.jpg` to `/var/www/html/` after every detection event and every timelapse tick. Open `http://<pi-address>/obj.jpg` in a browser to see the latest detection.

---

## Configuration

Create `~/.filchrc` (TOML format):

```toml
[global]
database  = "/home/pi/surveillance"   # where annotated JPEGs are archived
url       = "http://192.168.1.42"     # URL where the database folder above is visible in nginx

# Daylight-only operation: sleep between sunset and sunrise (default: true).
# Set to false to run around the clock (indoor cameras, IR-lit setups, etc.).
# latitude and longitude are required only when daylight_only = true.
daylight_only = true
latitude  = 55.6050                   # decimal degrees, used for sunrise/sunset
longitude = 13.0038                   # decimal degrees

# Optional tuning
timelapse_period = 600    # seconds between timelapse frames (default: 600)
sleep_time       = 2      # seconds to wait between detection cycles to reduce CPU load (default: 2)
dusk_delay       = 30     # minutes after sunset to keep running (default: 30)
jpg_quality      = 80     # JPEG quality for full size images (default: 80)
jpg_quality_web  = 75     # JPEG quality for web-previews (default: 75)
web_object_jpg    = "/var/www/html/obj.jpg"        # web-preview path for detection images
web_timelapse_jpg = "/var/www/html/timelapse.jpg"  # web-preview path for timelapse images

# Optional: path to a plain-text labels file, one label per line.
# Only needed when the model does not embed labels (rare).
# labels = "/path/to/coco_labels.txt"

[ntfy]
channel = "my-filch-alerts"   # ntfy.sh topic name; leave empty to disable

[filter]
# Both lists are optional. Omit a section to disable that filter.

# Suppress detections where ALL objects belong to this list.
ignore = ["bird", "car"]

# When set, only save/notify when at least one object is in this list.
follow = ["person", "cat"]
```

**Coordinates** — find your latitude/longitude on [latlong.net](https://www.latlong.net/) or from Google Maps (right-click → "What's here?").

**ntfy** — install the [ntfy app](https://ntfy.sh/) on your phone, subscribe to the channel name you create/chose, and notifications arrive instantly. No account needed.

---

## Running

```bash
./filch.py
```

The script starts the camera, loads the default model, and enters the detection loop. Stop it with `Ctrl+C` or `SIGTERM`.

### Command-line options

```
-c, --config PATH       Path to the TOML configuration file (default: ~/.filchrc)
--model PATH            Path to .rpk model file
                        (default: SSD MobileNetV2 from imx500-models)
--threshold FLOAT       Minimum detection confidence, 0–1 (default: 0.55)
--iou FLOAT             IoU threshold for NMS (default: 0.65)
--max-detections N      Discard lowest-confidence results above this count (default: 10)
--fps N                 Override camera frame rate
--labels PATH           Path to a labels file (overrides model-embedded labels)
--postprocess nanodet   Enable NanoDet post-processing for compatible models
--bbox-order xy|yx      Bounding box coordinate order (default: yx)
--bbox-normalization    Treat bounding boxes as normalised 0–1 coordinates
-r, --preserve-aspect-ratio  Maintain input tensor aspect ratio
--print-intrinsics      Print model network_intrinsics JSON and exit
```

### Running as a systemd service

```ini
# /etc/systemd/system/filch.service
[Unit]
Description=Filch AI surveillance
After=network.target

[Service]
ExecStart=/home/pi/filch/filch.py
User=pi
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now filch
sudo journalctl -fu filch    # follow logs
```

---

## Image archive layout

```
/home/pi/surveillance/
├── 250516/
│   ├── 250516-073412-142-person.jpg
│   ├── 250516-073412-142-person-cat.jpg
│   ├── 250516-080000-timelapse.jpg
│   └── ...
├── 250517/
│   └── ...
```

Filenames encode the timestamp: `YYMMDD-HHMMSS-mmm-label1-label2.jpg`. Timelapse frames omit milliseconds and end with `-timelapse.jpg`.

---

## Models

The default model (`imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk`) detects the [80 COCO classes](https://cocodataset.org/#explore) — people, vehicles, animals, furniture, and common household objects.

Additional models are available in the [Raspberry Pi IMX500 Model Zoo](https://github.com/raspberrypi/imx500-models). To use one:

```bash
./filch.py --model /usr/share/imx500-models/imx500_network_yolov8n_pp.rpk
```

Use `--print-intrinsics` to inspect any model's embedded metadata before deploying it.

---

## Acknowledgements

Built on top of the [Picamera2](https://github.com/raspberrypi/picamera2) library and inspired by the `imx500_object_detection_demo.py` example from the Raspberry Pi foundation.
