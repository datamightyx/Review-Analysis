"""Optional semantic candidate recall, preferring the locally fine-tuned model.

Model resolution order (first available wins):
  1. models/finetuned_minilm_static — model2vec distillation of the tuned
     model (pure numpy, instant load);
  2. models/finetuned_minilm — the contrastively fine-tuned transformer
     produced by scripts/train_embeddings.py from human gate_labels.json
     verdicts (loads torch; encode stays torch-only because this machine's
     torch<->numpy bridge is broken);
  3. minishlab/potion-base-8M — the static base model (pure numpy).

Both fallbacks keep the pipeline running when nothing was trained yet.

The embeddings are used ONLY to surface extra [similar: ...] candidates for
the LLM judge. With the BASE model raw cosine is anti-correlated with the
SOP merge rule ("Works well"~"Works great" = 0.87 must NOT merge, "They
worked outstandin"~"Works great" = 0.22 MUST merge) — the fine-tuned model
is trained to fix exactly that, but decisions still stay with
merge_compatible and the judge; better vectors only mean better candidate
recall. Cosine is good at recall either way: it finds paraphrases that
share no words ("these are the right tool for the job" ~ "Work well").

Degrades silently: if no backend is available the pipeline runs exactly as
before. Disable explicitly with SCORING_NO_EMBEDDINGS=1.
"""
from __future__ import annotations

import os
from pathlib import Path

_BASE_MODEL = "minishlab/potion-base-8M"
_MODELS_DIR = Path(__file__).parent.parent / "models"
_FINETUNED_ST = _MODELS_DIR / "finetuned_minilm"
_FINETUNED_STATIC = _MODELS_DIR / "finetuned_minilm_static"

_model = None            # object with .encode(list[str]) -> ndarray
_failed = False
_cache: dict[str, "object"] = {}


class _StaticBackend:
    """model2vec static model — already returns numpy arrays."""

    def __init__(self, name_or_path: str):
        from model2vec import StaticModel
        self._m = StaticModel.from_pretrained(name_or_path)

    def encode(self, texts: list[str]):
        return self._m.encode(texts)


class _TransformerBackend:
    """sentence-transformers model, encoded torch-side: this machine's torch
    was built against numpy 1.x (numpy 2.x installed), so tensor.numpy() is
    broken — go tensor -> lists -> fresh ndarray instead."""

    def __init__(self, path: str):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        self._np = np
        self._m = SentenceTransformer(path, device="cpu")

    def encode(self, texts: list[str]):
        emb = self._m.encode(texts, convert_to_tensor=True,
                             show_progress_bar=False)
        return self._np.array(emb.cpu().tolist(), dtype="float32")


def _load_model():
    if _FINETUNED_STATIC.exists():
        try:
            return _StaticBackend(str(_FINETUNED_STATIC))
        except Exception:
            pass
    if _FINETUNED_ST.exists():
        try:
            return _TransformerBackend(str(_FINETUNED_ST))
        except Exception:
            pass
    return _StaticBackend(_BASE_MODEL)


def available() -> bool:
    global _model, _failed
    if _failed or os.environ.get("SCORING_NO_EMBEDDINGS"):
        return False
    if _model is not None:
        return True
    try:
        _model = _load_model()
        return True
    except Exception:
        _failed = True
        return False


def _vectors(texts: list[str]):
    import numpy as np
    missing = [t for t in texts if t not in _cache]
    if missing:
        emb = _model.encode(missing)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        for t, v in zip(missing, emb):
            _cache[t] = v
    return np.stack([_cache[t] for t in texts])


def top_semantic(text: str, items: list[tuple[str, str]], k: int = 3,
                 min_cos: float = 0.35) -> list[tuple[str, str, float]]:
    """items: (id, text). Up to k semantically closest (id, text, cosine).
    Returns [] when embeddings are unavailable."""
    if not items or not available():
        return []
    try:
        vecs = _vectors([text] + [t for _, t in items])
    except Exception:
        return []
    query, rest = vecs[0], vecs[1:]
    scored = [(items[i][0], items[i][1], float(rest[i] @ query))
              for i in range(len(items))]
    scored = [s for s in scored if s[2] >= min_cos]
    scored.sort(key=lambda s: -s[2])
    return scored[:k]
