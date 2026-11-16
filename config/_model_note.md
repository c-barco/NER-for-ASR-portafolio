# Model choice per environment

| environment | model | why |
|---|---|---|
| Project laptop (Intel Core 5 120U, 15.6 GB RAM, no CUDA) | `Qwen/Qwen3-ASR-0.6B-hf` @ `7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c` (1.58 GB) | CPU-only; float32 1.7B needs ~8 GB RAM and is very slow. Laptop is for technical pilots only. |
| University GPU (JupyterHub/cluster used in the labs) | `Qwen/Qwen3-ASR-1.7B-hf` @ `bcd2b5b7f32b480ab5790554cfa8347f246a14f3` (4.08 GB) | Model used in lab_06 and the lecturer notebooks. |

The reported baseline must use ONE model for the whole evaluation. Decide it with the team before
the full run; the configs default to the lab model (1.7B) and the laptop pilot overrides to 0.6B.
Neither model has been run successfully on the laptop yet (see docs/continuity).
