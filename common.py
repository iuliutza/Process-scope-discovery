"""Shared helpers: cached engine calls, text snippets, report."""
from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime

import pandas as pd
import streamlit as st

import engine as E
import viz as V

HERE = os.path.dirname(os.path.abspath(__file__))
DEMO_LOG = os.path.join(HERE, "data", "logistics.json")
DEMO_GT = os.path.join(HERE, "data", "gt_logistics.json")

PROCELLAR_REPO = "https://github.com/hudsonjychen/procellar"
LOGISTICS_URL = "https://www.ocel-standard.org/event-logs/simulations/logistics/"

GLOSSARY = {
    "signature": "The set of object types an event touches, e.g. {Container, Truck}. Signatures are the nodes "
                 "of the graph; a scope is a group of signatures.",
    "hand-over": "An object's step from one event to its next event (time order), counted once per event pair. "
                 "Hand-overs between two different signatures are the edges.",
    "participants": "Jaccard distance of the two type sets: 1 − |shared| / |all|. High = the set of object types "
                    "changes a lot (other kinds of objects take part).",
    "multiplicity": "Median over hand-overs of 1 − min/max of the number of objects of a shared type in the two events; "
                    "max over shared types. High = batching or unbatching.",
    "wait": "1 − 1/r with r = median hand-over gap / the slower inside pace of the two sides; 0 if the hand-over "
            "is not slower than both. High = the object waits between the two pieces of work.",
    "pace_change": "|pace x − pace y| / max(pace x, pace y), inside paces of a shared type. High = the rhythm of work "
                   "changes without a pause. Off by default.",
    "b": "Boundary score of an edge: weighted mean of the four signals (0 = same piece of work, 1 = clear change).",
    "merge distance": "Steps-weighted mean b over all edges between two groups, both directions pooled.",
    "balancer": "λ: per signature pair, b' = b · (1 − λ · rec), rec = min/max of the hand-overs each way. "
                "Back-and-forth traffic counts as togetherness. 0 = off.",
    "markov time": "Infomap's zoom level: how long the random walker is assumed to stay before its moves are coded. "
                   "Small = many small modules, large = few big ones.",
}


def tip(key):
    return GLOSSARY.get(key, "")


def file_key(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()[:16]


@st.cache_resource(show_spinner="Reading the log and building timelines…", max_entries=4)
def get_log(key: str, _data: bytes | None = None, path: str | None = None, proc_qual="process"):
    L = E.load_log(path if path else _data, proc_qual=proc_qual)
    G = E.build_graph(L)
    return L, G


@st.cache_resource(show_spinner=False, max_entries=16)
def get_signals(key: str, theta: float, _L, _G):
    return E.signals(_L, _G, theta)


@st.cache_resource(show_spinner=False, max_entries=64)
def get_scores_merge(key: str, theta: float, weights: tuple, balancer: float, _L, _S):
    B = E.scores(_S, dict(weights), balancer)
    M = E.merge(_L, _S, B)
    return B, M


@st.cache_resource(show_spinner=False, max_entries=16)
def get_layout(key: str, theta: float, _L, _S):
    return V.layout(_L, _S.edges)


@st.cache_resource(show_spinner="Running Infomap…", max_entries=256)
def get_infomap(key: str, theta: float, weights: tuple, weight: str, mt: float, seed: int, trials: int, _L, _S, _B):
    return E.run_infomap(_L, _S, _B, mt, weight=weight, seed=seed, trials=trials)


@st.cache_resource(show_spinner="Applying the ground truth…", max_entries=8)
def get_gt(key: str, gt_key: str, _L, _gt):
    gts = E.gt_sets(_L, _gt)
    sev = E.sig_events(_L)
    return gts, sev, E.signature_gt_share(_L, gts, sev)


def demo_bundle():
    with open(DEMO_LOG, "rb") as f:
        data = f.read()
    key = "demo-" + file_key(data)
    L, G = get_log(key, None, DEMO_LOG)
    with open(DEMO_GT, encoding="utf-8") as f:
        gt = json.load(f)
    return key, L, G, gt


def wtuple(w):
    return tuple(sorted((k, float(v)) for k, v in w.items()))


# ---------------------------------------------------------------- comparisons
def event_ari(L, a: dict, b: dict):
    """Adjusted Rand index between two signature->scope maps, weighted by events."""
    n = Counter(); na = Counter(); nb = Counter()
    for s, c in L.n_events.items():
        if s in a and s in b:
            n[(a[s], b[s])] += c; na[a[s]] += c; nb[b[s]] += c
    N = sum(n.values())
    comb = lambda x: x * (x - 1) / 2
    sij = sum(comb(v) for v in n.values()); sa = sum(comb(v) for v in na.values()); sb = sum(comb(v) for v in nb.values())
    exp = sa * sb / comb(N) if N > 1 else 0
    mx = (sa + sb) / 2
    return 1.0 if mx == exp else (sij - exp) / (mx - exp)


def gt_majority(share):
    """signature -> (majority GT scope, share)."""
    out = {}
    for s, d in share.items():
        if d:
            g = max(d, key=d.get); out[s] = (g, d[g])
    return out


def scope_table(L, parts, ap=3):
    rows = []
    for i, g in enumerate(parts, 1):
        acts = Counter()
        for s in g:
            acts.update(L.sig_acts[s])
        rows.append({"scope": i, "events": sum(L.n_events[s] for s in g), "signatures": len(g),
                     "signature list": "; ".join(sorted(g, key=lambda s: -L.n_events[s])),
                     "main activities": ", ".join(a for a, _ in acts.most_common(ap))})
    return pd.DataFrame(rows)


def cut_edges(L, S, B, scope_of):
    rows = []
    for (x, y), c in S.edges.items():
        if scope_of.get(x) != scope_of.get(y):
            r = S.raw[(x, y)]
            rows.append({"from scope": scope_of.get(x), "to scope": scope_of.get(y), "from": x, "to": y, "hand-overs": c,
                         **{k: round(v, 3) for k, v in r.items()}, "b": round(B.b[(x, y)], 3)})
    return pd.DataFrame(rows).sort_values("hand-overs", ascending=False) if rows else pd.DataFrame()


def signal_overview(S, B):
    """Per signal: edges where it fires, share of hand-overs, spread; constant-signal warnings."""
    tot = sum(S.edges.values()) or 1
    rows, warn = [], []
    for k in E.SIGNALS:
        vals = [(S.raw[e][k], c) for e, c in S.edges.items()]
        fire = [(v, c) for v, c in vals if v > 1e-9]
        vs = [v for v, _ in vals]
        mean_w = sum(v * c for v, c in vals) / tot
        rows.append({"signal": V.SIGNAL_LABEL[k], "key": k, "edges where > 0": f"{len(fire)} / {len(vals)}",
                     "share of hand-overs": sum(c for _, c in fire) / tot, "min": min(vs) if vs else 0,
                     "max": max(vs) if vs else 0, "mean (hand-over weighted)": mean_w,
                     "spread (sd)": float(pd.Series(vs).std(ddof=0)) if vs else 0})
        if vs and max(vs) - min(vs) < 1e-9:
            warn.append(f"**{V.SIGNAL_LABEL[k]}** has the same value ({vs[0]:.2f}) on every edge, so its weight does not "
                        "change the merge order — it only shifts all b values.")
        elif vs and sum(1 for v in vs if v > 0.95) / len(vs) > 0.8:
            warn.append(f"**{V.SIGNAL_LABEL[k]}** is close to 1 on most edges (saturated); it barely tells edges apart.")
    df = pd.DataFrame(rows)
    corr = pd.DataFrame({V.SIGNAL_LABEL[k]: [S.raw[e][k] for e in S.edges] for k in E.SIGNALS}).corr().round(2) if len(S.edges) > 2 else None
    return df, corr, warn


# ---------------------------------------------------------------- report
def report_md(log_name, L, S, B, params, method, level, parts, rules_check, f1=None, ties=None):
    scope_of = E.scope_map(parts)
    lines = [f"# Scope proposal: {log_name}", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by the scope tool. The tool proposes scopes; naming and "
             "defining the processes stays with the analyst.", "",
             "## Log", "",
             f"- events: {len(L.ev_act):,} ({L.ev_sig.count(None):,} without objects → no scope)",
             f"- object types: {len(set(L.otype.values()))}, signatures: {len(L.n_events)}, "
             f"edges: {len(S.edges)} of {S.n_all_edges} (θ = {params['theta']:g})", ""]
    if L.n_bad_time:
        lines.append(f"- {L.n_bad_time} events with unreadable timestamps (left out of the timelines)")
    lines += ["## Parameters", "", f"- method: **{method}**, level: **{level}**",
              "- boundary score b: weighted mean; weights: " + ", ".join(f"{V.SIGNAL_LABEL[k]} {params['weights'][k]:g}" for k in E.SIGNALS),
              f"- distance balancer λ: {params['balancer']:g}"]
    if method == "Infomap":
        lines.append(f"- Infomap link weight: {params['weight']}, seed {params['seed']}, {params['trials']} trials, directed, top-level modules")
    lines += ["", f"## Scopes ({len(parts)})", ""]
    for i, g in enumerate(parts, 1):
        ev = sum(L.n_events[s] for s in g)
        lines.append(f"### scope {i} — {ev:,} events")
        for s in sorted(g, key=lambda s: -L.n_events[s]):
            acts = ", ".join(f"{a} ({n:,})" for a, n in L.sig_acts[s].most_common(4))
            lines.append(f"- `{s}` — {L.n_events[s]:,} events: {acts}")
        lines.append("")
    ce = cut_edges(L, S, B, scope_of)
    lines += ["## Borders: hand-overs between scopes", ""]
    if len(ce):
        lines.append("| from → to | hand-overs | participants | batching | wait | pace change | b |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in ce.to_dict("records"):
            lines.append(f"| {r['from']} (s{r['from scope']}) → {r['to']} (s{r['to scope']}) | {r['hand-overs']:,} | "
                         f"{r['participants']:.2f} | {r['multiplicity']:.2f} | {r['wait']:.2f} | {r['pace_change']:.2f} | {r['b']:.3f} |")
    else:
        lines.append("No hand-overs between scopes.")
    lines.append("")
    if f1 is not None:
        df, macro = f1
        lines += ["## Comparison with the ground truth", "", f"macro F1 = **{macro:.3f}** (best-matching scope per GT scope)", "",
                  "| GT scope | events | best scope | precision | recall | F1 |", "|---|---|---|---|---|---|"]
        for r in df.to_dict("records"):
            lines.append(f"| {r['GT scope']} | {r['events']:,} | {r['best scope']} | {r['precision']:.2f} | {r['recall']:.2f} | {r['F1']:.2f} |")
        lines.append("")
    lines += ["## Procellar rules", "",
              ("Check passed: every signature matches exactly its own scope (Procellar semantics)." if not rules_check else
               f"Check FAILED for {len(rules_check)} signatures: " + "; ".join(f"{s} (expected scope {sc}, matched {h})" for s, sc, h in rules_check)),
              "Rules use object types only (no activities); events without objects get no scope.", ""]
    notes = ["## Notes", ""]
    if ties and method == "Hierarchical merge":
        notes.append(f"- Ties in the merge at steps {', '.join(map(str, ties))}: equal distances are broken by key order (signature name order).")
    notes.append("- A scope may cut an object's life on purpose; objects can cross scopes.")
    notes.append("- Scopes are horizontal (sequential) splits only; case-type variants are not detected.")
    return "\n".join(lines + notes) + "\n"
