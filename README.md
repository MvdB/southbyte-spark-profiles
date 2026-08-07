# southbyte-spark-profiles

Hardware-validated serving profiles for the **NVIDIA DGX Spark** (GB10 SoC,
sm_120, 128 GB unified memory, aarch64). This is the one repo in the
[southbyte](https://github.com/MvdB?tab=repositories&q=southbyte) family that is
deliberately hardware-specific — the frameworks that consume these profiles
(`southbyte-vllm`, `southbyte-tts`) are hardware-agnostic and read their tuning
from here.

## Layout

```
vllm/
├── profiles/   # one <owner>--<model-name>/vllm_profile.conf per model (52),
│               #   hand-validated PROFILE_* settings for GB10 / 128 GB
└── custom/     # Dockerfiles for models needing sm_120 kernels not in
                #   stock vllm/vllm-openai (NVFP4 etc.) + patch_conv3d.py
vllm-omni/
└── voxtral_tts_stages.yaml   # GB10-tuned vLLM-omni stage config for Voxtral TTS
                              #   (reduced gpu_mem_util for co-residency;
                              #   enforce_eager — sm_120 CUDA-graph produced
                              #   garbled audio)
```

## How the frameworks use it

- **`southbyte-vllm`** — `runner/vllm_spark.sh` serves from each model's
  `vllm_profile.conf` in `~/hf_models/<model>/`. The curated copies here are the
  source of truth: copy `vllm/profiles/<model>/` into `~/hf_models/<model>/` to
  pin known-good settings, or let the profiler auto-generate. Custom images build
  from `vllm/custom/` (e.g. `docker build -t spark-mistral-small4:v1 -f vllm/custom/Dockerfile.mistral-small4 .`).
- **`southbyte-tts`** — `serving/run_voxtral_tts.sh` mounts
  `vllm-omni/voxtral_tts_stages.yaml` via `--stage-configs-path`. Path is
  overridable with `SPARK_PROFILES_DIR` (default `~/southbyte/southbyte-spark-profiles`).

## Naming convention

Model directories use `<owner>--<model-name>` (HF id with `/` → `--`), defined by
[southbyte-sync](https://github.com/MvdB/southbyte-sync).

## License

MIT — see [LICENSE](LICENSE)
