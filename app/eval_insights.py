"""Conclusions for the eval dashboard, computed from the loaded snapshots.

The dashboard captions used to be hard-coded ("Hybrid wins on all 3 corpora", "only
Vector scales", "model size dominates") and silently went stale whenever a re-run
changed the numbers. These pure functions derive every statement from the data and
say how strongly the data backs it: a gap counts as significant only if the
snapshot's paired test says so (p < 0.05 and a CI excluding 0) or, for snapshots
without a paired test, if the two 95% CIs are disjoint; snapshots with no uncertainty
at all (older runs) are flagged as such.

No streamlit / pandas import: unit-testable in the light CI job.
"""

from collections import Counter

ALPHA = 0.05
_VERDICT = {True: "significant", False: "within noise", None: "significance unknown"}


def short_name(name: str) -> str:
    """Vector / Hybrid / Graph from a stack name (display name, short or lowercase key)."""
    n = name.lower()
    return "Vector" if "vecto" in n else ("Hybrid" if "hybr" in n else "Graph")


def fmt_ci(ci) -> str:
    """"[lo, hi]" with 3 decimals, or "—" when the snapshot has no interval."""
    return f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "—"


def _fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _join(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def _every(n: int) -> str:
    return "on every dataset" if n > 1 else "on this dataset"


def oriented_pair(paired: dict | None, a: str, b: str) -> dict | None:
    """The paired test of a - b from a snapshot's "paired" block (stored either way)."""
    if not paired:
        return None
    if f"{a}-{b}" in paired:
        return paired[f"{a}-{b}"]
    if f"{b}-{a}" in paired:
        r = paired[f"{b}-{a}"]
        return r | {"mean_diff": -r["mean_diff"], "lo": -r["hi"], "hi": -r["lo"]}
    return None


def is_significant(pair: dict, alpha: float = ALPHA) -> bool:
    """Paired test rejects "no difference": p < alpha AND the CI of the diff excludes 0."""
    return pair["p_value"] < alpha and (pair["lo"] > 0 or pair["hi"] < 0)


def gap_evidence(a_ci, b_ci, pair: dict | None) -> tuple[bool | None, str]:
    """(significant?, why) for the gap between two stacks.

    Prefers the paired test; otherwise disjoint 95% CIs (a conservative check:
    overlapping CIs do not prove the absence of a difference); otherwise None.
    """
    if pair is not None:
        return is_significant(pair), f"paired p={_fmt_p(pair['p_value'])}"
    if a_ci and b_ci:
        disjoint = a_ci[0] > b_ci[1] or b_ci[0] > a_ci[1]
        return disjoint, "95% CIs " + ("disjoint" if disjoint else "overlap")
    return None, "no CI in this snapshot"


def beir_caption(loaded: dict[str, dict], metric: str = "ndcg@10") -> str:
    """Leader per dataset and whether its lead over the runner-up is significant."""
    leaders, parts = [], []
    for ds, snap in loaded.items():
        stacks = {short_name(k): v for k, v in snap["stacks"].items()}
        first, second = sorted(stacks, key=lambda s: stacks[s][metric], reverse=True)[:2]
        a, b = stacks[first], stacks[second]
        sig, why = gap_evidence(a.get("ci95", {}).get(metric), b.get("ci95", {}).get(metric),
                                oriented_pair(snap.get("paired"), first, second))
        leaders.append(first)
        parts.append(f"**{ds}**: {first} {a[metric]:.3f} vs {second} {b[metric]:.3f} "
                     f"({_VERDICT[sig]}, {why})")
    if len(set(leaders)) == 1:
        lead = f"{leaders[0]} ranks first {_every(len(leaders))}"
    else:
        lead = "The leader changes by dataset"
    return f"{lead} — {'; '.join(parts)}."


def rerank_caption(loaded: dict[str, dict]) -> str:
    """Per dataset: which reranking mode gains more, and on how many stacks significantly."""
    winners, parts = [], []
    for ds, snap in loaded.items():
        stacks = list(snap["stacks"].values())
        mean = {mode: sum(s[f"delta_{mode}"] for s in stacks) / len(stacks)
                for mode in ("replace", "fusion")}
        best = max(mean, key=mean.get)
        other = "fusion" if best == "replace" else "replace"
        winners.append(best)
        text = f"**{ds}**: {best} {mean[best]:+.3f} vs {other} {mean[other]:+.3f}"
        tests = [s["paired"][f"delta_{best}"] for s in stacks if "paired" in s]
        if tests:
            text += f" (Δ significant for {sum(map(is_significant, tests))}/{len(tests)} stacks)"
        parts.append(text)
    if len(set(winners)) == 1:
        lead = f"*{winners[0]}* gains more {_every(len(winners))}"
    else:
        lead = "The better mode changes by dataset → measure per dataset"
    return f"{lead} (mean Δ nDCG@10 over the stacks) — {'; '.join(parts)}."


def throughput_changes(stacks: dict, lo: int = 1, hi: int = 4) -> dict[str, float]:
    """Relative throughput change from `lo` to `hi` threads, per stack, best first."""
    changes = {}
    for name, m in stacks.items():
        qps = m.get("throughput_qps", {})
        if str(lo) in qps and str(hi) in qps and qps[str(lo)]:
            changes[short_name(name)] = qps[str(hi)] / qps[str(lo)] - 1
    return dict(sorted(changes.items(), key=lambda kv: kv[1], reverse=True))


def throughput_caption(stacks: dict, lo: int = 1, hi: int = 4, gain: float = 0.2) -> str:
    """Which stacks gain throughput with more threads — read off the data."""
    changes = throughput_changes(stacks, lo, hi)
    if not changes:
        return ""
    scaling = [s for s, c in changes.items() if c >= gain]
    detail = ", ".join(f"{s} {c:+.0%}" for s, c in changes.items())
    who = (f"{_join(scaling)} gain{'s' if len(scaling) == 1 else ''} ≥ {gain:.0%}" if scaling
           else f"no stack gains ≥ {gain:.0%}")
    return f"From {lo} to {hi} threads: {detail} — {who}."


def build_caption(build_seconds: dict) -> str:
    """The costliest index to build, from the measured build times."""
    totals = {s: build_seconds.get(f"{s.lower()}_total") for s in ("Vector", "Hybrid", "Graph")}
    totals = {s: t for s, t in totals.items() if t is not None}
    if not totals:
        return ""
    worst = max(totals, key=totals.get)
    text = f"{worst} is the costliest index to build ({totals[worst]:.0f}s)"
    if worst == "Graph" and build_seconds.get("graph_ner_build") is not None:
        text += f", of which spaCy NER ≈ {build_seconds['graph_ner_build']:.0f}s"
    return text + "."


def run_label(snap: dict, fallback: str) -> str:
    """Name of an answer-eval run: its model (+ prompt variant when not the default)."""
    cfg = snap.get("config", {})
    model = cfg.get("model")
    label = model if model and model != "?" else fallback
    prompt = cfg.get("prompt")
    return f"{label} ({prompt} prompt)" if prompt and prompt != "default" else label


def label_runs(snaps: dict[str, dict]) -> dict[str, dict]:
    """{label: snapshot} from {file stem: snapshot}, keeping one entry per file.

    Two runs with the same model and prompt (a re-run with another n or k) get the
    same `run_label`; keyed by it, one would silently overwrite the other, so colliding
    labels get their file stem appended.
    """
    labels = {stem: run_label(snap, stem) for stem, snap in snaps.items()}
    counts = Counter(labels.values())
    return {(label if counts[label] == 1 else f"{label} · {stem}"): snaps[stem]
            for stem, label in labels.items()}


def run_setup(snap: dict) -> tuple[int | None, str]:
    """(questions, prompt variant) of an answer-eval run. Snapshots from before
    `--prompt` existed record no prompt: they all used the app's default one."""
    cfg = snap.get("config", {})
    n = cfg.get("n_queries") or next(iter(snap["stacks"].values()), {}).get("n_queries")
    return n, cfg.get("prompt") or "default"


def answer_caption(runs: dict[str, dict], metric: str = "f1") -> str:
    """Mean score per run (model) + the largest gap between architectures within each run,
    with the caveats that keep runs from being ranked like-for-like when they are not."""
    if not runs:
        return ""
    means = {label: sum(s[metric] for s in snap["stacks"].values()) / len(snap["stacks"])
             for label, snap in runs.items()}
    setups = {label: run_setup(snap) for label, snap in runs.items()}
    ranked = sorted(means, key=means.get, reverse=True)
    parts = [f"Mean {metric.upper()} over the stacks: " + ", ".join(
        f"{label} {means[label]:.3f}" + (f" (n={setups[label][0]})" if setups[label][0] else "")
        for label in ranked)]
    if len(set(setups.values())) > 1:
        parts.append("The runs differ in sample size or prompt: not a like-for-like ranking")
    verbose = [label for label in ranked if setups[label][1] == "default"]
    if verbose and metric in ("em", "f1"):
        parts.append(f"{_join(verbose)} used the default prompt, which lets the model answer "
                     f"in full sentences: there {metric.upper()} mostly measures verbosity "
                     "(a correct but wordy answer scores low) — compare *contains* where "
                     "recorded, or runs with `--prompt short`")
    gaps = []
    for label in ranked:
        snap = runs[label]
        stacks = {short_name(k): v for k, v in snap["stacks"].items()}
        best, worst = (max(stacks, key=lambda s: stacks[s][metric]),
                       min(stacks, key=lambda s: stacks[s][metric]))
        n = setups[label][0]
        gap = stacks[best][metric] - stacks[worst][metric]
        if not gap:
            gaps.append(f"{label}: all stacks equal")
            continue
        pair = (oriented_pair(snap.get("paired"), best, worst)
                if snap.get("paired_metric", metric) == metric else None)
        sig, why = gap_evidence(stacks[best].get("ci95", {}).get(metric),
                                stacks[worst].get("ci95", {}).get(metric), pair)
        gaps.append(f"{label}: {best} − {worst} = {gap:+.3f} "
                    f"({_VERDICT[sig]}, {why}{f', n={n}' if n else ''})")
    parts.append("Largest gap between architectures — " + "; ".join(gaps))
    return ". ".join(parts) + "."


def provenance_note(config: dict) -> str:
    """" · code <sha> (+ uncommitted changes)" from a snapshot's provenance, or "" if absent."""
    prov = config.get("provenance") or {}
    sha = prov.get("git_sha")
    if not sha:
        return ""
    return f" · code {sha[:7]}" + (" + uncommitted changes" if prov.get("git_dirty") else "")
