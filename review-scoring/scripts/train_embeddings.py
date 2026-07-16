"""Fine-tune a local embedding model on the human merge verdicts.

Raw cosine similarity contradicts the SOP merge rules ("Works well" ~
"Works great" scores ~0.87 but must NEVER merge; "They worked outstandin" ~
"Works great" scores ~0.22 but MUST merge). This script teaches a small
sentence-transformer the actual rules with contrastive learning:

  positive pair  = a pair the human labeled ✗ «злити» (label: merge) in the
                   «🕵 Перевірити» tab — the two wordings are one message;
  negative pair  = a pair the human labeled ✓ «лишити» (label: keep) — the
                   wordings must stay separate rows.

Labels are pooled from EVERY product folder's gate_labels.json (same pool
the gate precedents use). On top of them, a built-in set of UNIVERSAL pairs
encodes the domain-agnostic SOP grammar itself (praise tiers never mix,
qualifiers never swap, one-sided negation splits) — disable with
--no-sop-pairs.

The tuned model is saved to models/finetuned_minilm and picked up
automatically by pipeline/embeddings.py on the next run (candidate recall
for the judge only — merge DECISIONS stay with the gate and the judge).
When gate_labels are missing or fewer than --min-labels, nothing is saved
and the pipeline keeps using the base static model.

Environment note: this machine's torch was built against numpy 1.x while
numpy 2.x is installed, so the torch<->numpy bridge is broken. Everything
here deliberately stays in torch (manual training loop, tokenize-based
collate, convert_to_tensor) and never touches HF datasets/accelerate.

Usage:
    python scripts/train_embeddings.py
    python scripts/train_embeddings.py --epochs 12 --min-labels 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.precedents import aggregate_all_labels
from pipeline.similarity import normalize

DEFAULT_BASE = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OUT = ROOT / "models" / "finetuned_minilm"

# Universal SOP-grammar pairs (no product vocabulary — only generic
# evaluative wordings that mean the same in any category). 1.0 = same
# message per the SOP, 0.0 = must stay separate rows.
SOP_PAIRS: list[tuple[str, str, float]] = [
    # plain praise tier + effectiveness idioms are ONE message
    ("works good", "works well", 1.0),
    ("it really works", "works well", 1.0),
    ("does the trick", "works well", 1.0),
    ("gets the job done", "works well", 1.0),
    ("the right tool for the job", "works well", 1.0),
    ("they work", "works well", 1.0),
    # strong praise tier is ONE message
    ("works great", "work perfectly", 1.0),
    ("these work excellent", "works great", 1.0),
    ("worked fantastic", "works amazing", 1.0),
    ("works flawlessly", "works perfect", 1.0),
    # intensifiers / subjects / typos never split
    ("works very well", "works really well", 1.0),
    ("this works so well", "works well", 1.0),
    ("it worked great for her", "works great", 1.0),
    ("they're gonna work great", "works great", 1.0),
    ("works amazin", "works amazing", 1.0),
    ("work perfectl", "works perfectly", 1.0),
    # negated complaints merge across negation form and degree words
    ("they just don't work", "doesn't work at all", 1.0),
    ("didn't work very well", "they just don't work", 1.0),
    ("it's not working", "has no effect", 1.0),
    ("stopped working", "quit working", 1.0),
    ("easy to use", "simple to use", 1.0),
    # praise tiers NEVER mix
    ("works well", "works great", 0.0),
    ("works good", "works perfectly", 0.0),
    ("works good", "works excellent", 0.0),
    ("worked ok", "works well", 0.0),
    ("worked ok", "works great", 0.0),
    ("works fine", "works amazing", 0.0),
    # qualifiers are content and never swap, even for synonyms
    ("works fast", "works quickly", 0.0),
    ("works quickly", "works instantly", 0.0),
    ("helps right away", "helps instantly", 0.0),
    ("stopped it quickly", "stopped it immediately", 0.0),
    ("lasted a day", "lasted a week", 0.0),
    ("came off in seconds", "came off after a day", 0.0),
    # one-sided negation always splits
    ("works well", "doesn't work well", 0.0),
    ("easy to use", "not easy to use", 0.0),
    ("holds up well", "doesn't hold up", 0.0),
    # a long multi-detail sentence never dissolves into a short generic row
    ("works well", "worked well for the first day but then quit and had to "
     "be replaced with a different one", 0.0),
]


def load_label_pairs(products_root: Path) -> list[tuple[str, str, float]]:
    pairs = []
    for rec in aggregate_all_labels(products_root).values():
        a = (rec.get("phrase") or "").strip()
        b = (rec.get("into") or "").strip()
        lab = rec.get("label")
        if a and b and lab in ("merge", "keep"):
            pairs.append((a, b, 1.0 if lab == "merge" else 0.0))
    return pairs


def build_dataset(products_root: Path, use_sop: bool,
                  min_labels: int, force: bool
                  ) -> list[tuple[str, str, float]]:
    labeled = load_label_pairs(products_root)
    print(f"Мітки людини (gate_labels.json, всі продукти): {len(labeled)} пар")
    if len(labeled) < min_labels and not force:
        sys.exit(
            f"Замало міток ({len(labeled)} < {min_labels}) — модель НЕ "
            f"навчається, пайплайн лишається на базовій моделі.\n"
            f"Розмітьте більше пар у вкладці «🕵 Перевірити» або запустіть "
            f"з --force / --min-labels.")
    seen = {(normalize(a), normalize(b)) for a, b, _ in labeled}
    seen |= {(normalize(b), normalize(a)) for a, b, _ in labeled}
    extra = []
    if use_sop:
        extra = [(a, b, l) for a, b, l in SOP_PAIRS
                 if (normalize(a), normalize(b)) not in seen]
        print(f"Універсальні SOP-пари: +{len(extra)}")
    data = labeled + extra
    pos = sum(1 for _, _, l in data if l == 1.0)
    print(f"Разом {len(data)} пар: {pos} «злити» / {len(data) - pos} «лишити»")
    return data


def cos_pairs(model, pairs) -> list[float]:
    """Cosine per pair, all-torch (the numpy bridge is broken here)."""
    import torch
    with torch.no_grad():
        a = model.encode([p[0] for p in pairs], convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
        b = model.encode([p[1] for p in pairs], convert_to_tensor=True,
                         normalize_embeddings=True, show_progress_bar=False)
        return (a * b).sum(dim=1).tolist()


def report(tag: str, model, data) -> None:
    scores = cos_pairs(model, data)
    pos = [s for s, (_, _, l) in zip(scores, data) if l == 1.0]
    neg = [s for s, (_, _, l) in zip(scores, data) if l == 0.0]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    # the single best threshold separating «злити» from «лишити»
    best_acc = 0.0
    for t in [i / 100 for i in range(-20, 101, 2)]:
        acc = (sum(1 for s in pos if s >= t) + sum(1 for s in neg if s < t)) \
              / max(len(scores), 1)
        best_acc = max(best_acc, acc)
    print(f"{tag}: cos(злити)={mean(pos):.3f}  cos(лишити)={mean(neg):.3f}  "
          f"розрив={mean(pos) - mean(neg):+.3f}  "
          f"точність@найкращий поріг={best_acc:.2%}")
    demo = cos_pairs(model, [("Works well", "Works great", 0.0),
                             ("They worked outstandin", "Works great", 1.0)])
    print(f'{tag}: "Works well"~"Works great" (різні рівні) = {demo[0]:.3f}; '
          f'"worked outstandin"~"Works great" (одне) = {demo[1]:.3f}')


def train(model, data, epochs: int, batch_size: int, lr: float,
          margin: float, seed: int = 42) -> None:
    import torch
    from torch.utils.data import DataLoader
    from sentence_transformers import losses

    # margin is the target cosine DISTANCE for «лишити» pairs: they keep
    # incurring loss until cos <= 1 - margin. The SOP negatives share most
    # of their words with the positives ("works well"/"works great"), so a
    # lax margin lets the positive pull win — hence the high default.
    loss_model = losses.ContrastiveLoss(model=model, margin=margin)

    prep = getattr(model, "preprocess", None) or model.tokenize

    def collate(batch):
        feats = [prep([a for a, _, _ in batch]),
                 prep([b for _, b, _ in batch])]
        labels = torch.tensor([l for _, _, l in batch])
        return feats, labels

    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    dl = DataLoader(data, batch_size=batch_size, shuffle=True,
                    generator=gen, collate_fn=collate)
    opt = torch.optim.AdamW(loss_model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for feats, labels in dl:
            opt.zero_grad()
            loss = loss_model(feats, labels)
            loss.backward()
            opt.step()
            total += float(loss)
        print(f"  епоха {epoch + 1}/{epochs}: loss={total / len(dl):.4f}")
    model.eval()


def try_distill(out_dir: Path) -> None:
    """Best effort: distill the tuned transformer into a model2vec static
    model — pure-numpy at inference, so the pipeline avoids importing torch
    on every run. On this machine the torch->numpy bridge is broken, so
    distillation may fail; the pipeline then loads the transformer itself."""
    static_dir = out_dir.parent / (out_dir.name + "_static")
    try:
        from model2vec.distill import distill
        m = distill(model_name=str(out_dir), pca_dims=256)
        m.save_pretrained(str(static_dir))
        print(f"Статичну (швидку) версію збережено: {static_dir}")
    except Exception as e:
        print(f"Статична дистиляція недоступна ({type(e).__name__}: "
              f"{str(e)[:120]}) — пайплайн вантажитиме трансформер напряму.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Донавчання локальної embedding-моделі на вердиктах "
                    "gate_labels.json (contrastive learning)")
    ap.add_argument("--products-root", default=str(ROOT / "products"))
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--margin", type=float, default=0.8,
                    help="цільова косинусна відстань для пар «лишити»")
    ap.add_argument("--min-labels", type=int, default=10,
                    help="мінімум людських міток, щоб навчання мало сенс")
    ap.add_argument("--no-sop-pairs", action="store_true",
                    help="навчатися лише на людських мітках")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    data = build_dataset(Path(args.products_root), not args.no_sop_pairs,
                         args.min_labels, args.force)

    print(f"Завантаження базової моделі {args.base} …")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.base, device="cpu")

    report("ДО  ", model, data)
    train(model, data, args.epochs, args.batch, args.lr, args.margin)
    report("ПІСЛЯ", model, data)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir))
    print(f"Модель збережено: {out_dir}")
    try_distill(out_dir)
    print("Готово — наступний прогін пайплайна підхопить модель автоматично "
          "(pipeline/embeddings.py).")


if __name__ == "__main__":
    main()
