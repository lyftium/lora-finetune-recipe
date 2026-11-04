"""Fail closed if loaders start trusting remote code casually."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "lora_recipe"


def test_no_trust_remote_code_in_loaders():
    offenders = []
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        if "trust_remote_code=True" in text or 'trust_remote_code = True' in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert not offenders, f"trust_remote_code=True forbidden: {offenders}"


def test_model_ingest_doc_exists():
    assert (ROOT / "docs" / "model-ingest.md").is_file()
