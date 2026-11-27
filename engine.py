"""
Scope engine: the thesis notebook cells as functions.

  load_log      signatures, timelines                        (merge cell, step 1-2)
  build_graph   steps, gaps, multiplicity per edge           (merge cell, step 2)
  signals       four signals per edge, noise filter theta    (merge cell, step 3-5)
  scores        boundary score b (weighted mean) + balancer  (merge cell, step 5)
  merge         hierarchical merge, cut at k or tau          (merge cell, step 6-7)
  run_infomap   Infomap on the signature graph               (infomap cell)
  make_rules    Procellar rules + Procellar-semantics check  (stage 5 cell)
  gt_sets/score GT event sets and macro F1                   (f1 curve cell)

Only change against the notebook: COMBINE is fixed to "mean" in the tool.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median

import pandas as pd

SIGNALS = ["participants", "multiplicity", "wait", "pace_change"]
DEFAULTS = dict(theta=0.0,
                weights={"participants": 1.0, "multiplicity": 1.0, "wait": 1.0, "pace_change": 0.0},
                balancer=0.0, proc_qual="process",
                weight="together", markov_times=[0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0],
                seed=42, trials=20)


# ---------------------------------------------------------------- 1. log + signatures
@dataclass
class Log:
    otype: dict
    ev_act: list
    ev_objs: list
    t: object
    bad: object
    ev_sig: list
    sig_types: dict
    n_events: Counter
    sig_acts: dict
    n_bad_time: int = 0

    @property
    def sigs(self):
        return sorted(self.n_events)

    def label(self, s, n=3):
        return f"{s} [{', '.join(a for a, _ in self.sig_acts[s].most_common(n))}]"


def load_log(src, proc_qual="process") -> Log:
    """src: path, file-like, bytes or already parsed dict (OCEL 2.0 JSON)."""
    if isinstance(src, dict):
        log = src
    elif isinstance(src, (bytes, bytearray)):
        log = json.loads(src)
    elif hasattr(src, "read"):
        log = json.load(src)
    else:
        with open(src, encoding="utf-8") as f:
            log = json.load(f)
    for key in ("objects", "events"):
        if key not in log:
            raise ValueError(f"not an OCEL 2.0 JSON file: key '{key}' missing")
    otype = {o["id"]: o["type"] for o in log["objects"]}
    ev_act, ev_time, ev_objs = [], [], []
    for e in log["events"]:
        ev_act.append(e["type"]); ev_time.append(e["time"])
        ev_objs.append(list(dict.fromkeys(r["objectId"] for r in e.get("relationships") or []
                                          if r.get("qualifier") != proc_qual and r["objectId"] in otype)))
    ts = pd.to_datetime(pd.Series(ev_time), utc=True, errors="coerce", format="mixed")
    bad = ts.isna().to_numpy(); t = ts.values.astype("int64")
    sig_types, ev_sig = {}, []
    for objs in ev_objs:
        if objs:
            ty = tuple(sorted({otype[o] for o in objs})); s = "{" + ", ".join(ty) + "}"
            sig_types[s] = set(ty); ev_sig.append(s)
        else:
            ev_sig.append(None)
    n_events = Counter(s for s in ev_sig if s is not None)
    sig_acts = defaultdict(Counter)
    for i, s in enumerate(ev_sig):
        if s is not None:
            sig_acts[s][ev_act[i]] += 1
    return Log(otype, ev_act, ev_objs, t, bad, ev_sig, sig_types, n_events, dict(sig_acts), int(bad.sum()))


# ---------------------------------------------------------------- 2. graph + timelines
@dataclass
class Graph:
    steps: Counter
    gaps_edge_type: dict
    pace_inside: dict
    mult_edge: dict
    pair_acts: dict
    obj_seq: dict = field(repr=False)


def build_graph(L: Log) -> Graph:
    ev_sig, ev_objs, otype, t, bad = L.ev_sig, L.ev_objs, L.otype, L.t, L.bad
    obj_seq = defaultdict(list)
    for i in sorted(range(len(L.ev_act)), key=lambda i: (t[i], i)):
        if not bad[i] and ev_sig[i] is not None:
            for o in ev_objs[i]:
                obj_seq[o].append(i)
    steps, seen_pair = Counter(), set()
    gaps_edge_type, gaps_inside = defaultdict(list), defaultdict(list)
    seen_pair_type = set()
    mult_edge = defaultdict(list)
    pair_acts = defaultdict(Counter)
    for o, seq in obj_seq.items():
        T = otype[o]
        for a, b_ in zip(seq, seq[1:]):
            x, y = ev_sig[a], ev_sig[b_]
            if (a, b_, T) not in seen_pair_type:
                seen_pair_type.add((a, b_, T))
                g = int(t[b_] - t[a])
                if x == y and g > 0:
                    gaps_inside[(x, T)].append(g)
                if x != y:
                    gaps_edge_type[(x, y, T)].append(g)
            if x != y and (a, b_) not in seen_pair:
                seen_pair.add((a, b_)); steps[(x, y)] += 1
                pair_acts[(x, y)][(L.ev_act[a], L.ev_act[b_])] += 1
                ca = Counter(otype[o2] for o2 in ev_objs[a]); cb = Counter(otype[o2] for o2 in ev_objs[b_])
                for T2 in ca.keys() & cb.keys():
                    mult_edge[(x, y, T2)].append(1 - min(ca[T2], cb[T2]) / max(ca[T2], cb[T2]))
    pace_inside = {k: median(v) for k, v in gaps_inside.items() if v}
    return Graph(steps, dict(gaps_edge_type), pace_inside, dict(mult_edge), dict(pair_acts), dict(obj_seq))


# ---------------------------------------------------------------- 3-4. noise filter + signals
@dataclass
class Signals:
    edges: dict          # (x, y) -> steps (after theta)
    raw: dict            # (x, y) -> {signal: value}
    why: dict            # (x, y) -> {signal: deciding type / ratio text}
    n_all_edges: int


def signals(L: Log, G: Graph, theta=0.0) -> Signals:
    out_s, in_s = Counter(), Counter()
    for (x, y), c in G.steps.items():
        out_s[x] += c; in_s[y] += c
    edges = {e: c for e, c in G.steps.items() if c / out_s[e[0]] >= theta or c / in_s[e[1]] >= theta}

    def top(vals):
        return max(vals, key=lambda vr: vr[0]) if vals else (0.0, "-")

    raw, why = {}, {}
    for (x, y), c in edges.items():
        st = L.sig_types
        shared = st[x] & st[y]
        part = 1 - len(shared) / len(st[x] | st[y])
        mvals = [(median(G.mult_edge[(x, y, T)]), T) for T in sorted(shared) if G.mult_edge.get((x, y, T))]
        mult, mult_t = top(mvals)
        tvals = []
        for T in sorted(shared):
            g = G.gaps_edge_type.get((x, y, T))
            refs = [p for p in (G.pace_inside.get((x, T)), G.pace_inside.get((y, T))) if p]
            if g and refs:
                r = median(g) / max(refs)
                tvals.append((1 - 1 / r if r > 1 else 0.0, f"{T} x{r:.1f}"))
        tim, tim_t = top(tvals)
        pvals = []
        for T in sorted(shared):
            px, py = G.pace_inside.get((x, T)), G.pace_inside.get((y, T))
            if px and py:
                pvals.append((abs(px - py) / max(px, py), T))
        pch, pch_t = top(pvals)
        raw[(x, y)] = {"participants": part, "multiplicity": mult, "wait": tim, "pace_change": pch}
        why[(x, y)] = {"participants": ", ".join(sorted(shared)) or "-", "multiplicity": mult_t,
                       "wait": tim_t, "pace_change": pch_t}
    return Signals(edges, raw, why, len(G.steps))


# ---------------------------------------------------------------- 5. boundary score (mean only)
@dataclass
class Scores:
    b: dict
    b_merge: dict
    rec: dict


def scores(S: Signals, weights: dict, balancer=0.0) -> Scores:
    assert 0.0 <= balancer <= 1.0
    assert all(weights.get(k, 0) >= 0 for k in SIGNALS) and sum(weights.get(k, 0) for k in SIGNALS) > 0
    wsum = sum(weights.get(k, 0) for k in SIGNALS)
    b, bm, rec = {}, {}, {}
    for (x, y), c in S.edges.items():
        score = sum(weights.get(k, 0) * v for k, v in S.raw[(x, y)].items()) / wsum
        back = S.edges.get((y, x), 0)
        r = min(c, back) / max(c, back)
        b[(x, y)] = score; rec[(x, y)] = r
        bm[(x, y)] = score * (1 - balancer * r)
    return Scores(b, bm, rec)


def edge_table(L: Log, S: Signals, B: Scores) -> pd.DataFrame:
    rows = []
    for (x, y), c in S.edges.items():
        r, w = S.raw[(x, y)], S.why[(x, y)]
        rows.append({"from": x, "to": y, "steps": c,
                     "participants": r["participants"], "multiplicity": r["multiplicity"],
                     "wait": r["wait"], "pace_change": r["pace_change"],
                     "b": B.b[(x, y)], "rec": B.rec[(x, y)], "b balanced": B.b_merge[(x, y)],
                     "multiplicity type": w["multiplicity"], "wait (type, x pace)": w["wait"],
                     "pace change type": w["pace_change"]})
    return pd.DataFrame(rows).sort_values("steps", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- 6-7. hierarchical merge
@dataclass
class Merge:
    sigs: list
    merges: list         # (d, p, q, new_key)
    node: dict
    tops: list           # keys of the unconnected top groups
    ties: list           # merge steps where the smallest distance was tied

    def leaves(self, k):
        n = self.node[k]
        return [n[1]] if n[0] == "leaf" else self.leaves(n[1]) + self.leaves(n[2])

    def cut(self, n_merges):
        part = {k: frozenset([s]) for k, s in enumerate(self.sigs)}
        for d, p, q, k in self.merges[:n_merges]:
            part[k] = part.pop(p) | part.pop(q)
        return list(part.values())

    def n_merges_for(self, k=None, tau=None):
        if k is not None:
            return max(0, min(len(self.merges), len(self.sigs) - k))
        return sum(1 for d, *_ in self.merges if d <= tau)


def merge(L: Log, S: Signals, B: Scores) -> Merge:
    edges, bm = S.edges, B.b_merge

    def dist(A, Bg):
        es = [(edges[e], bm[e]) for e in edges if (e[0] in A and e[1] in Bg) or (e[0] in Bg and e[1] in A)]
        if not es:
            return None
        return sum(c * v for c, v in es) / sum(c for c, _ in es)

    sigs = L.sigs
    groups = {k: frozenset([s]) for k, s in enumerate(sigs)}
    node = {k: ("leaf", next(iter(g))) for k, g in groups.items()}
    nxt, merges, ties = len(groups), [], []
    while len(groups) > 1:
        best, cands = None, []
        keys = sorted(groups)
        for i, p in enumerate(keys):
            for q in keys[i + 1:]:
                d = dist(groups[p], groups[q])
                if d is None:
                    continue
                cands.append(d)
                if best is None or d < best[0]:
                    best = (d, p, q)
        if best is None:
            break
        d, p, q = best
        if sum(1 for c in cands if abs(c - d) < 1e-12) > 1:
            ties.append(len(merges) + 1)
        groups[nxt] = groups.pop(p) | groups.pop(q)
        node[nxt] = ("merge", p, q, d)
        merges.append((d, p, q, nxt))
        nxt += 1
    return Merge(sigs, merges, node, sorted(groups, key=lambda k: -len(groups[k])), ties)


def by_size(L: Log, parts):
    """Scopes numbered by size (events), as in the notebook."""
    return sorted((frozenset(g) for g in parts), key=lambda g: -sum(L.n_events[s] for s in g))


def scope_map(parts):
    return {s: i for i, g in enumerate(parts, 1) for s in g}


# ---------------------------------------------------------------- Infomap
def run_infomap(L: Log, S: Signals, B: Scores, mt, weight="together", seed=42, trials=20):
    from infomap import Infomap
    sig_list = L.sigs
    sid = {s: i for i, s in enumerate(sig_list)}
    im = Infomap(silent=True, directed=True, markov_time=mt, seed=seed, num_trials=trials)
    for s in sig_list:
        im.add_node(sid[s])
    for (x, y), c in S.edges.items():
        w = c * (1 - B.b[(x, y)]) if weight == "together" else c
        if w > 0:
            im.add_link(sid[x], sid[y], w)
    im.run()
    mods = defaultdict(list)
    for node_id, module in im.get_modules(depth_level=1).items():
        mods[module].append(sig_list[node_id])
    return by_size(L, mods.values()), im.codelength


# ---------------------------------------------------------------- stage 5: Procellar rules
def _rule(proc, name, inc_ot=(), exc_ot=()):
    return {"ruleName": name, "parentProcess": proc,
            "includeOT": {"entities": sorted(inc_ot), "condition": []},
            "includeAct": {"entities": [], "condition": []},
            "excludeOT": {"entities": sorted(exc_ot), "condition": []},
            "excludeAct": {"entities": [], "condition": []}}


def _tree(op, nodes):
    t = nodes[0]
    for n in nodes[1:]:
        t = {"op": op, "left": t, "right": n}
    return t


def rule_match(r, E, act=None):
    io, ia = r["includeOT"]["entities"], r["includeAct"]["entities"]
    eo, ea = r["excludeOT"]["entities"], r["excludeAct"]["entities"]
    inc = True if not (io or ia) else ((bool(io) and all(o in E for o in io)) or (bool(ia) and act in ia))
    exc = (bool(eo) and all(o in E for o in eo)) or (bool(ea) and act in ea)
    return inc and not exc


def proc_match(p, E, act=None):
    rules = {r["ruleName"]: r for r in p["rules"]}

    def ev(n):
        if isinstance(n, str):
            return rule_match(rules[n], E, act)
        l, r = ev(n["left"]), ev(n["right"])
        return (l and r) if n["op"] == "and" else (l or r)
    return ev(p["relations"]) if p["relations"] else any(rule_match(r, E, act) for r in p["rules"])


def make_rules(L: Log, src: dict):
    """src: signature -> scope number. Returns (processes, bad_sigs)."""
    sigs = {s: frozenset(L.sig_types[s]) for s in src}
    processes = []
    for sc in sorted(set(src.values())):
        proc = f"scope {sc}"
        members = sorted((s for s in src if src[s] == sc), key=lambda s: (len(sigs[s]), s))
        chosen = []
        for s in members:
            S_ = sigs[s]
            if any(S0 <= S_ and not (S_ & X0) for S0, X0 in chosen):
                continue
            foreign = [sigs[e] for e in src if src[e] != sc and sigs[e] > S_]
            X, left = set(), list(foreign)
            while left:
                cnt = Counter(t for E in left for t in E - S_)
                tbest = sorted(cnt, key=lambda t: (-cnt[t], t))[0]
                X.add(tbest)
                left = [E for E in left if tbest not in E]
            chosen.append((S_, frozenset(X)))
        rules, nodes, not_rules = [], [], {}
        for k, (S_, X) in enumerate(chosen, 1):
            name = f"s{k}"
            if len(X) <= 1:
                rules.append(_rule(proc, name, S_, X)); nodes.append(name)
            else:
                rules.append(_rule(proc, name, S_))
                parts = [name]
                for t in sorted(X):
                    if t not in not_rules:
                        not_rules[t] = f"not {t}"
                        rules.append(_rule(proc, not_rules[t], (), [t]))
                    parts.append(not_rules[t])
                nodes.append(_tree("and", parts))
        processes.append({"processName": proc, "imported": False, "rules": rules,
                          "relations": _tree("or", nodes)})
    bad = []
    for s, E in sigs.items():
        hits = [p["processName"] for p in processes if proc_match(p, E)]
        if hits != [f"scope {src[s]}"]:
            bad.append((s, src[s], hits))
    return processes, bad


# ---------------------------------------------------------------- GT + F1
def gt_sets(L: Log, gt_procs):
    sets = {p["processName"]: set() for p in gt_procs}
    for i in range(len(L.ev_act)):
        E = {L.otype[o] for o in L.ev_objs[i]}
        for p in gt_procs:
            if proc_match(p, E, L.ev_act[i]):
                sets[p["processName"]].add(i)
    return sets


def sig_events(L: Log):
    out = {}
    for i, s in enumerate(L.ev_sig):
        if s is not None:
            out.setdefault(s, set()).add(i)
    return out


def _prf(g, a):
    tp = len(g & a)
    p = tp / len(a) if a else 0.0
    r = tp / len(g) if g else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(groups, gts, sev):
    """Per GT scope: best-matching found scope by F1 (many-to-one). Returns (table, macro F1)."""
    pred = [set().union(*(sev[s] for s in g)) for g in groups]
    rows = []
    for name, gs in sorted(gts.items()):
        best = max(((_prf(gs, a), i) for i, a in enumerate(pred, 1)), key=lambda x: x[0][2],
                   default=((0, 0, 0), None))
        (p, r, f), i = best
        rows.append({"GT scope": name, "events": len(gs), "best scope": i,
                     "precision": p, "recall": r, "F1": f})
    df = pd.DataFrame(rows)
    return df, (float(df["F1"].mean()) if len(df) else 0.0)


def signature_gt_share(L: Log, gts, sev):
    """signature -> {GT scope: share of its events}."""
    return {s: {g: len(ev & gs) / len(ev) for g, gs in gts.items() if ev & gs} for s, ev in sev.items()}
