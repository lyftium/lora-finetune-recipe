# Model ingest policy

Rules for loading base models in this recipe. Prefer failure over silent risk.

1. **Pin a revision** — configs set `revision:` (default `main` only for the tiny CPU smoke model).
2. **Prefer safetensors** — rely on Hugging Face defaults; do not force pickle loads.
3. **No casual `trust_remote_code=True`** — loaders in `src/lora_recipe/model.py` must not pass it. CI: `tests/test_model_ingest_policy.py`.
4. **QLoRA refuses CPU** — clear error instead of silent fallback (`QLoraUnavailable`).
5. **Lineage** — every `train_run.json` records git sha, config hash, data hashes, library versions.

Interview note: employer air-gap policies may additionally require mirrored registries and signed artefacts; this repo demonstrates the engineering habits, not a certified supply chain.
