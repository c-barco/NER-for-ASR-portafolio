# Continuity - Step 1-2 preparation (2026-11-16)

## Objective and authorisation
Prepare the shared Step 1-2 pipeline (dataset comparison, baseline WER/CER evaluation) while dataset
approval is pending. Scope was narrowed during the session to Steps 1-2 only. The user chose
"code only for now": no model download or inference on the laptop.
Not authorised and not done: commit, push, emails, paid services.

## Repository
* Remote: https://github.com/c-barco/NER-for-ASR-portafolio (private; viewer has WRITE)
* Checkout: `C:\Users\arthu\Desktop\NER-for-ASR-portafolio` (fresh clone; `main` had only README, LICENSE, .gitignore)
* Branch: `step2-baseline-pipeline` (local only, **uncommitted**). `main` is untouched.
* No AGENTS.md or contribution rules in the repository; no branch protection on `main`.

## Environment (verified)
* Windows 11 Home 10.0.26200; Intel Core 5 120U (10 cores / 12 threads); 15.6 GB RAM; Intel integrated
  graphics, **no CUDA**; C: 284 GB free.
* Python 3.14.7 (only interpreter). Project venv: `.venv` (git-ignored). Jupyter was not installed globally.
* **Windows Smart App Control is ON** and blocks unsigned DLLs: `pyarrow._compute` (→ pandas, datasets),
  `numpy.random`, spaCy (ujson), and the `jupyter-nbconvert.exe` launcher. Not changed (a security setting
  that is the user's decision; it may be irreversible).
* Kernel `ner-asr` was registered, but into the app-virtualised AppData path; the user should re-run the
  `ipykernel install` command from the README in their own terminal.

## Checks performed (real results)
* `pytest`: **24 passed** (metrics incl. lab_06 slide example, normalisation, duration selection,
  insufficient-audio error, split mixing refusal, manifest round-trip, incremental save, crash/resume,
  failure recording + retry, transcriber receives only audio paths, export columns, config validation,
  end-to-end CLI on fixture). All with SYNTHETIC fixtures / a fake transcriber.
* Notebook executed headless (nbclient) with `config/pilot_fixture.yaml`: all cells ran; outputs are
  FIXTURE numbers only.
* Lecturer notebooks in Materials are byte-identical to GitHub `jwirth-iisys/mmai_06@924b10a`.
* HF metadata (2026-11-16): Qwen3-ASR-0.6B-hf 1.58 GB @7f1569a; 1.7B-hf 4.08 GB @bcd2b5b; both
  Apache-2.0, not gated. VoxPopuli EN test ≈ 4.93 h, MLS EN test ≈ 15.55 h (viewer statistics).
* **Not run:** any real model inference, any HF dataset streaming, Common Voice, the cluster path.

## Blockers / open questions
See `docs/open_questions.md` (dataset approval, ~40 h meaning, deadline/format, compute).

## Exact next action
1. When approval arrives: choose the config, run the pilot on the university JupyterHub (Linux, GPU) with
   `NER_ASR_RUN_INFERENCE=1`, inspect the outputs, then decide the full-run duration.
2. Review the diff and commit the branch (user's decision).
