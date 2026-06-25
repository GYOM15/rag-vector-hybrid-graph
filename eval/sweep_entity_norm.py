"""Balayage de la normalisation du boost entité (graphe), validé en *held-out*.

On choisit la forme de normalisation sur des splits de **validation**
(SciFact-train, NFCorpus-dev), puis on rapporte sur les splits de **test**
intouchés (SciFact-test, HotpotQA-val, NFCorpus-test). Aucune fuite : le test ne
sert jamais à la sélection. L'index/graphe est bâti **une seule fois** par corpus ;
on ne fait que remplacer la fonction de normalisation (le scoring est à la requête).

    python -m eval.sweep_entity_norm
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from shared.ir_metrics import ndcg_at_k  # noqa: E402
from pipeline import STACK_NAMES, assemble_stacks  # noqa: E402
from stack3_graphrag.retriever import _ENTITY_NORMS  # noqa: E402

from eval.beir_eval import _ranked_doc_ids, K_MAX, load_beir, load_hotpot_distractor  # noqa: E402

NORMS = list(_ENTITY_NORMS)  # ordre : none, log, p25, sqrt, p75, linear


def graph_ndcg(retr, norm_fn, queries_eval, qrels) -> float:
    """nDCG@10 moyen du graphe pour une normalisation donnée (swap à chaud)."""
    retr._norm = norm_fn
    total = 0.0
    for qid, qtext in queries_eval:
        ranked = _ranked_doc_ids(retr.search(qtext, k=K_MAX))
        total += ndcg_at_k(ranked, qrels[qid], 10)
    return total / max(1, len(queries_eval))


def graph_retriever(texts, metadata):
    """Construit les stacks (1 fois) et renvoie le retriever graphe."""
    return assemble_stacks(texts, metadata)[STACK_NAMES["graph"]].retriever


def main() -> None:
    # (label, role, retriever, queries, qrels) — role ∈ {val, test}
    evals = []

    print("Chargement + indexation SciFact…", flush=True)
    texts, meta, q_test, qr_test = load_beir("scifact", "test")
    _, _, q_train, qr_train = load_beir("scifact", "train")
    retr = graph_retriever(texts, meta)
    evals.append(("SciFact-train", "val", retr, q_train, qr_train))
    evals.append(("SciFact-test", "test", retr, q_test, qr_test))

    print("Chargement + indexation NFCorpus…", flush=True)
    texts, meta, q_test, qr_test = load_beir("nfcorpus", "test")
    _, _, q_val, qr_val = load_beir("nfcorpus", "validation")
    retr = graph_retriever(texts, meta)
    evals.append(("NFCorpus-val", "val", retr, q_val, qr_val))
    evals.append(("NFCorpus-test", "test", retr, q_test, qr_test))

    print("Chargement + indexation HotpotQA…", flush=True)
    texts, meta, q_hp, qr_hp = load_hotpot_distractor(500)
    retr = graph_retriever(texts, meta)
    evals.append(("HotpotQA-val", "test", retr, q_hp, qr_hp))

    # nDCG[label][norm]
    results: dict[str, dict[str, float]] = {}
    for label, role, retr, queries, qrels in evals:
        print(f"\n== {label} ({role}, {len(queries)} requêtes) ==", flush=True)
        results[label] = {}
        for norm in NORMS:
            val = graph_ndcg(retr, _ENTITY_NORMS[norm], queries, qrels)
            results[label][norm] = val
            print(f"   {norm:7s} nDCG@10 = {val:.4f}", flush=True)

    # Sélection sur validation uniquement
    val_labels = [lbl for lbl, role, *_ in evals if role == "val"]
    test_labels = [lbl for lbl, role, *_ in evals if role == "test"]
    mean_val = {n: sum(results[lbl][n] for lbl in val_labels) / len(val_labels) for n in NORMS}
    best = max(NORMS, key=lambda n: mean_val[n])

    print("\n" + "=" * 64)
    print("VALIDATION (held-out) — moyenne nDCG@10 sur", val_labels)
    for n in sorted(NORMS, key=lambda n: mean_val[n], reverse=True):
        star = "  <== choisi" if n == best else ""
        print(f"   {n:7s} {mean_val[n]:.4f}{star}")

    print(f"\nNorme retenue : '{best}'  →  report sur TEST (intouché) :")
    header = "   " + "".join(f"{lbl:16s}" for lbl in test_labels)
    print(header)
    print("   " + "".join(f"{results[lbl][best]:<16.4f}" for lbl in test_labels))
    print("\n   (comparaison) 'none' (bug) vs 'sqrt' (actuel) vs choisi, sur test :")
    for n in ("none", "sqrt", best):
        print(f"   {n:7s} " + "".join(f"{results[lbl][n]:<16.4f}" for lbl in test_labels))
    print("\n>>> FIN")


if __name__ == "__main__":
    main()
