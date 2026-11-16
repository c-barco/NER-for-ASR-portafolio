# Unresolved assignment requirements

Professor email already sent (dataset preferences: VoxPopuli EN > MLS EN > Common Voice EN).
Do not send another one from this list; use it to check the reply and for the next lab session.

1. **Dataset approval** - which dataset is approved as primary? Which is the second dataset for Step 8?
2. **~40 hours of audio** - does it apply to the whole project dataset (incl. fine-tuning data) or to the
   baseline evaluation? Official test splits: VoxPopuli EN ≈ 4.9 h, MLS EN ≈ 15.6 h. Reaching 40 h of
   *evaluation* audio would need other splits. Keep `selection.target_hours: null` until clarified.
3. **"At least 1,000 audio items"** (overview p. 23) - stated for datasets prepared for *fine-tuning*.
   It is an item count, separate from the hours question.
4. **Step 2 deadline and format** - Monday 23 Nov 2026 per our understanding; time of day, submission
   channel and expected format (notebook? PDF report? Git link?) are not confirmed.
5. **Model** - is any audio language model acceptable, or Qwen3-ASR-1.7B-hf as in lab_06? Is
   comparing 0.6B vs 1.7B wanted?
6. **Compute** - may we use the university JupyterHub / cluster GPUs and the `/training-1` data copies
   for the portfolio evaluation? (The laptop is CPU-only and Smart App Control blocks some libraries.)
7. **Data source** - for the approved dataset, should we use the cluster's Kaldi copy (lowercase
   labels for VoxPopuli) or the public Hugging Face version (cased `raw_text`)?
8. **Team** - overview p. 23 says teams of three by default; confirm team members and who submits.
9. **Learning diary** - overview p. 21 asks for roughly weekly entries (8-10 in the semester); p. 20
   lists personal learning reflections (4x) as portfolio part 9. Confirm which applies to the portfolio.
