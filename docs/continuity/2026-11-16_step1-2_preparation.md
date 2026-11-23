# Continuity update — 2026-11-22

The earlier exploratory dataset/model decision has been superseded. The active
configuration is EuroSpeech (`uk` English, `germany` German) with IBM Granite 4
Speech. The repository now includes a reusable Step 2 parquet export that
creates one bilingual artifact with exactly 1,000 EN and 1,000 DE rows. It does
not download data or run inference until its explicit command is invoked.

Current scope remains limited to data preparation and a no-context baseline.
The next deliberate action is to run the parquet export on an environment where
`datasets` and PyArrow work, inspect its printed validation report and submit
the resulting lightweight reference-only file. Model inference remains a separate
explicit action.
