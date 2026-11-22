# Fixed model choice

The project uses one model for English and German baseline experiments:

```text
ibm-granite/granite-4.0-1b-speech
revision: bd87ab862416353633ea431fe49b1614003623c5
```

It was selected for strong expected ASR performance in the project languages
and for its documented keyword-biasing mechanism, which can be evaluated later
for names and acronyms. This is a rationale, not a claimed improvement: no
real-model result is reported until the fixed no-context baseline has run.

The implementation follows the Granite 4 Speech model card with
`AutoProcessor` and `AutoModelForSpeechSeq2Seq`. Its no-context baseline request
is exactly `<|audio|>can you transcribe the speech into a written format?`.
`prompt=None` produces that request without a keyword suffix. If a later,
authorised experiment supplies a prompt, the wrapper treats it as a keyword
list and uses Granite's documented ` Keywords: ...` format.

Granite 4 Speech requires `transformers>=4.52.1` and a compatible PEFT
installation for its bundled adapter. The pinned project `transformers` version
is newer than that minimum. No model weights have been downloaded or run by
this repository update.
