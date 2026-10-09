# 模型与 wheel 清单

这些文件不进 git（`.gitignore`），打包时由 `scripts/build.sh` 从 `PAYLOAD_ROOTS` 指向的目录取（目录内按 `models/...`、`wheels/...` 的相对路径存放）。哈希与 `app/manifest.json` 一致，packager 会逐个校验。

| 文件 | 字节 | SHA256 |
|---|---:|---|
| `models/yolo11n_pose_rawhead_int8.rknn` | 4691086 | `3629dd1fe6c2229276295ff1050cf4a2cef9892bc87e4d2b50f5bef15bcc8add` |
| `models/asr/silero_vad.onnx` | 2327524 | `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3` |
| `models/asr/sensevoice_rv1126b_t172_scaled_hybrid.rknn` | 246399355 | `4f73132adf3235ed32152617df16c69d8581db5f1bf1d71278302bf6e2d015fb` |
| `models/asr/am.mvn` | 11203 | `29b3c740a2c0cfc6b308126d31d7f265fa2be74f3bb095cd2f143ea970896ae5` |
| `models/asr/embedding.npy` | 35968 | `83cf1fc5680fdf6d7edb411be5ce351cad4eca03b29a5bf5050aa19dfcc12267` |
| `models/asr/chn_jpn_yue_eng_ko_spectok.bpe.model` | 377341 | `aa87f86064c3730d799ddf7af3c04659151102cba548bce325cf06ba4da4e6a8` |
| `wheels/kaldi_native_fbank-1.22.3-cp311-cp311-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | 299054 | `d48e5dd8e897bf4509be2c6eeb4bbab728eaaef1f214ae0510c96219c4253d17` |
| `wheels/sentencepiece-0.2.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 1259182 | `27f90c55a65013cbb8f4d7aab0599bf925cde4adc67ae43a0d323677b5a1c6cb` |
| `wheels/sherpa_onnx-1.13.5-cp311-cp311-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | 4166120 | `dfedcebfe90d5079b98231e0d8ab008460c52cd44199698b464569c90dfd2124` |
| `wheels/sherpa_onnx_core-1.13.5-py3-none-manylinux2014_aarch64.whl` | 13051193 | `4cd751063a378a49f0c72eba5ba959fe375397f5baf93a53f3db64097d00e2aa` |
| `wheels/voxedge-0.0.9a0-py3-none-any.whl` | 280356 | `1af3abb53e149cda35ee637c574393e6e9f383b9c8993be90c8fe6fde17d93d7` |

## SenseVoice T172 模型来历

- 源：`sense-voice-encoder.scaled.fixed.onnx`（K=8 激活缩放，最后一个 FFN 的 `w_2` 除以 8、残差补 `Div(8)`，防 fp16 溢出），固定为 `speech[1,172,560]`。
- 转换：rknn-toolkit2 2.3.2，`target_platform="rv1126b"`，w4a16 为主，`tp_encoders`（后 20 块）与 CTC 头保持 fp16（混合精度）。
- 转换工作目录（WSL）：`/home/harve/sv_rknn/work/optimization-20261009/`；参数、字节数与 SHA：`~/.cache/recamera-build/asr-eval/optimization-20261009/conversion-parameters.tsv`（spark）。
- 当前 payload 位置（spark）：`~/.cache/recamera-build/asr-eval/optimization-20261009/t172/package-inputs/`。
