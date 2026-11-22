# Remaining decisions

The two core choices are resolved:

* Dataset: EuroSpeech (`uk` / English and `germany` / German).
* Model: IBM Granite 4 Speech (`ibm-granite/granite-4.0-1b-speech`).

No unresolved decision blocks the required Step 2 parquet export. Before a
later baseline run, the team may still decide:

1. Whether to run English, German, or both baselines first.
2. The explicit duration target for a full baseline evaluation; the code refuses
   to invent one when `--mode full` is selected.
3. The GPU environment and responsible submitter.

These choices do not authorise Steps 3–8 and are not prerequisites for the
dataset artifact.
