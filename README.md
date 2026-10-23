# Eldercare Monitor for reCamera Pro

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Rockchip%20RV1126B-green.svg)](https://www.seeedstudio.com/)

A multimodal on-device eldercare safety application designed for **reCamera Pro** (Rockchip RV1126B). It concurrently runs real-time multi-person pose estimation for fall detection on the NPU and continuous offline voice distress recognition via an acoustic pipeline in a single self-contained process.

---

## Key Features

- **Edge Multimodal AI**:
  - **Vision**: YOLO11n-Pose (17 COCO body keypoints) quantized in INT8 running on the 3.0 TOPS RV1126B NPU.
  - **Speech**: Continuous offline Automatic Speech Recognition (SenseVoice-Small T172 scaled hybrid precision) with Silero VAD on audio streams.
- **Privacy by Design**:
  - All video frames and audio chunks are processed zero-copy in local RAM.
  - No raw video, images, or audio clips leave the device or upload to external clouds.
- **Immediate Distress Alerts**:
  - Multiframe temporal state machine for fall detection (filtering out sitting, tying shoes, or bending down).
  - Emergency keyword detection ("Help", "救命", "摔倒了", "快来人") with immediate priority triggering.
- **Comprehensive Connectivity**:
  - Real-time skeleton visualization overlay plugin for the device Web Console.
  - Structured MQTT alert events compatible with Home Assistant and local IoT brokers.

---

## Performance Benchmarks

*Evaluated live on reCamera Pro running pose detection and offline ASR concurrently:*

| Metric | Measurement |
|---|---|
| ASR Chinese Character Error Rate (ZH CER) | **1.8%** (short segments) / **2.5%** (long segments) |
| ASR English Word Error Rate (EN WER) | **3.9%** (short segments) / **2.8%** (long segments) |
| ASR Real-Time Factor (RTF) | **≤ 0.11** (long speech segments) |
| Vision Fall Detection Inference | **15 - 20 FPS** (NPU only) |
| Concurrent Sustained Frame Rate | **~5.8 FPS** (continuous speech decoding) |
| Available System Memory | **> 500 MiB** (MemAvailable floor) |
| Power Consumption | **~3.5 W** (full dual-model load) |

---

## Architecture Overview

```text
 ┌──────────────────────┐         ┌──────────────────────┐
 │ Camera Sensor (VI)   │         │ Microphone (ALSA)    │
 └──────────┬───────────┘         └──────────┬───────────┘
            │ NV12 Frame                     │ 16kHz PCM
            ▼                                ▼
 ┌──────────────────────┐         ┌──────────────────────┐
 │ YOLO11n-Pose (RKNN)  │         │ Silero VAD (ONNX)    │
 └──────────┬───────────┘         └──────────┬───────────┘
            │ 17 Keypoints                   │ Active Speech
            ▼                                ▼
 ┌──────────────────────┐         ┌──────────────────────┐
 │ Fall State Machine   │         │ SenseVoice ASR (RKNN)│
 └──────────┬───────────┘         └──────────┬───────────┘
            │ Fall Event                     │ Distress Words
            └───────────────┬────────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Output Dispatcher         │
              │ - Web Console Overlay (WS)│
              │ - MQTT Event Dispatch     │
              └───────────────────────────┘
```

---

## Directory Structure

```text
eldercare-monitor/
├── app/
│   ├── app.py                     # Main application entry point and event loop
│   ├── asr_t172.py                # SenseVoice T172 single-model backend (window 172, hop 32)
│   ├── manifest.json              # App Center packaging manifest and resource claims
│   ├── icon.png                   # Application icon
│   ├── web/
│   │   └── overlay.html           # Live camera browser skeleton overlay plugin
│   ├── test_eldercare_monitor.py  # Unit and integration test suite
│   └── test_asr_t172.py           # ASR acoustic pipeline unit tests
├── scripts/
│   └── build.sh                   # Package assembly script
├── MODELS.md                      # Model inventory, hashes, and quantization lineage
├── LICENSE                        # Apache 2.0 license
└── README.md                      # This document
```

---

## Configuration & MQTT Protocol

The application supports flexible configuration via App Center or deployment orchestrators.

### Configuration Schema

```json
{
  "output_channels": ["ws", "mqtt"],
  "dMqtt": {
    "sURL": "192.168.1.50",
    "iPort": 1883,
    "sTopic": "living_room_camera",
    "sUsername": "",
    "sPassword": "",
    "sClientId": "recamera-pro-eldercare"
  }
}
```

### Event Payload Schema (`<sTopic>/eldercare-monitor/events`)

```json
{
  "timestamp": 1729676400.123,
  "device_id": "living_room_camera",
  "fall_alert": {
    "detected": true,
    "state": "fallen",
    "person_id": 1,
    "confidence": 0.94,
    "torso_angle": 78.5,
    "immobility_sec": 3.2
  },
  "voice_alert": {
    "detected": true,
    "text": "救命",
    "keyword": "救命",
    "confidence": 0.98,
    "duration_sec": 0.8
  }
}
```

---

## Building and Packaging

### Prerequisites

- Linux x86_64 host with Python 3.11+
- reCamera Pro SDK / kit repository (`recamera_pro`)

### Run Tests

```bash
export RECAMERA_PRO_REPO=/path/to/recamera_pro
uv run --frozen python -m pytest -q app/
```

### Build Package

```bash
./scripts/build.sh --test
# Outputs signed package: dist/eldercare-monitor-<version>-arm64.tar.gz
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
