"""Create tests/fixtures/kaldi_test (SYNTHETIC FIXTURE) for config/pilot_fixture.yaml."""

from pathlib import Path

from conftest import make_kaldi_fixture

if __name__ == "__main__":
    out = make_kaldi_fixture(Path(__file__).parent / "fixtures" / "kaldi_test")
    print(f"Fixture written to {out}")
