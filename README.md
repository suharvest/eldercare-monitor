# eldercare-monitor

reCamera Pro（RV1126B）养老监护应用：一个进程同时跑多人姿态跌倒检测（YOLO11n-Pose，NPU）和连续离线语音转写（silero VAD + SenseVoice），带浏览器叠加层插件。从 [recamera-pro-ext-api](https://github.com/Seeed-Studio/recamera-pro-ext-api) 的 `apps/eldercare-monitor`（`f080af5`）独立出来。

运行依赖设备上的 recamera_pro kit 与 appmgr；打包用 recamera_pro 的 `market/packaging`。

## 目录

| 路径 | 内容 |
|---|---|
| `app/` | 应用源码、`manifest.json`、叠加层 `web/overlay.html`、测试 |
| `app/asr_t172.py` | SenseVoice T172 单模型后端（窗口 172 帧、重叠 32、只取窗口中心） |
| `scripts/build.sh` | 打包 |
| `MODELS.md` | 模型 / wheel 清单与 T172 模型来历 |

## 测试

```bash
cd ~/project/recamera/recamera_pro
RECAMERA_PRO_REPO=$PWD uv run --frozen python -m pytest -q -p no:cacheprovider ~/project/eldercare-monitor/app
```

## 打包与签名

```bash
scripts/build.sh --test          # 产物：dist/eldercare-monitor-<version>-arm64.tar.gz
```

签名用 release 私钥（只在 Mac `~/.recamera_release_key/release_priv.pem`）：把包拷到 Mac 后运行 recamera_pro 的 `market/packaging/sign.py --dist <dir> --pkg <name>`，再 `--verify`。设备上通过 appmgr `POST /api/appMgr/install`（`path` + `signature`）安装。

## ASR 实测（2026-10-10，设备 live，姿态同时运行）

| 模型 | EN WER（短 / 长） | ZH CER（短 / 长） |
|---|---|---|
| 未量化参考 | 3.9% / 2.6% | 1.6% / 2.1% |
| 旧 w4a16 双模型（0.1.0） | 27.2% / 25.2% | 11.7% / 21.9% |
| T172 混合精度（0.1.5） | 3.9% / 2.8% | 1.8% / 2.5% |

语料 LibriSpeech test-clean 100 段 + AISHELL-1 test 100 段。0.1.5：短段解码 p50 0.53 s，长段 RTF ≤ 0.11，MemAvailable 最低 505 MiB。ASR 连续解码时姿态帧率从 6.0 降到约 3.5 fps；30 分钟正常运行均值 5.83 fps。
