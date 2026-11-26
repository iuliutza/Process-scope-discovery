"""Workbench: load an OCEL, explore signals, run both methods, compare, export rules + report."""
import json
import os

import pandas as pd
import streamlit as st

import common as C
import engine as E
import viz as V

D = E.DEFAULTS
st.title("Run on my log")

# ---------------------------------------------------------------- data source
with st.sidebar:
    st.header("Log")
    src = st.radio("Source", ["Upload an OCEL 2.0 JSON", "Use the Logistics demo log"], label_visibility="collapsed")
    up = None
    if src.startswith("Upload"):
        up = st.file_uploader("OCEL 2.0 JSON", type=["json", "jsonocel"])
    with st.expander("Ground truth (optional)"):
        gt_up = st.file_uploader("GT Procellar rules (JSON)", type=["json"], key="gt_up",
                                 help="A Procellar definition file. Used only to compute F1 against your own scopes.")
    proc_qual = D["proc_qual"]          # links to existing process objects are ignored (as in the notebook)

if src.startswith("Upload") and up is None:
    st.info("Upload an OCEL 2.0 JSON file in the sidebar, or switch to the Logistics demo log. "
            "The file stays on this machine.", icon=":material/upload_file:")
    st.stop()

try:
    if up is not None:
        data = up.getvalue()
        key = C.file_key(data) + proc_qual
        L, G = C.get_log(key, data, None, proc_qual)
        log_name = os.path.splitext(up.name)[0]
    else:
        key, L, G, _ = C.demo_bundle()
        log_name = "logistics"
except Exception as ex:  # noqa: BLE001
    st.error(f"Could not read the log: {ex}")
    st.stop()

gt, gt_key = None, None
if gt_up is not None:
    try:
        gt = json.loads(gt_up.getvalue()); gt_key = C.file_key(gt_up.getvalue())
    except Exception as ex:  # noqa: BLE001
        st.sidebar.error(f"GT file not readable: {ex}")
elif up is None:
    with open(C.DEMO_GT, encoding="utf-8") as f:
        gt = json.load(f)
    gt_key = "demo-gt"
    st.sidebar.caption(f"Demo: the [Logistics]({C.LOGISTICS_URL}) log; its documented phases are used as GT.")

if len(L.n_events) < 2:
    st.error("The log has fewer than two signatures; there is nothing to split.")
    st.stop()

# ---------------------------------------------------------------- parameters
with st.sidebar:
    st.header("Boundary score")
    w = {}
    for k in E.SIGNALS:
        w[k] = st.slider(V.SIGNAL_LABEL[k], 0.0, 2.0, float(D["weights"][k]), 0.1, key=f"wb_w_{k}", help=C.tip(k))
    if sum(w.values()) == 0:
        st.warning("All weights are 0; using equal weights.")
        w = dict(D["weights"])
    st.caption("b = weighted mean of the signals.")
    with st.expander("Advanced"):
        theta = st.number_input("Noise filter θ", 0.0, 0.5, float(D["theta"]), 0.01,
                                help="Keep edge x→y if its hand-overs are ≥ θ of x's outgoing or y's incoming. 0 = every hand-over counts.")
        lam = st.slider("Distance balancer λ (merge only)", 0.0, 1.0, float(D["balancer"]), 0.1, help=C.tip("balancer"))
        seed = st.number_input("Infomap seed", 0, 10_000, D["seed"])
        trials = st.number_input("Infomap trials", 1, 200, D["trials"])
        mts_txt = st.text_input("Markov times to sweep", ", ".join(f"{m:g}" for m in D["markov_times"]))
    try:
        mts = sorted({float(x) for x in mts_txt.replace(";", ",").split(",") if x.strip()})
        assert mts and all(m > 0 for m in mts)
    except Exception:  # noqa: BLE001
        st.sidebar.error("Markov times: comma-separated positive numbers.")
        mts = list(D["markov_times"])

S = C.get_signals(key, theta, L, G)
if not S.edges:
    st.error("No hand-overs between different signatures: the graph has no edges.")
    st.stop()
WT = C.wtuple(w)
B, M = C.get_scores_merge(key, theta, WT, lam, L, S)
B_plain = B if lam == 0 else C.get_scores_merge(key, theta, WT, 0.0, L, S)[0]
POS = C.get_layout(key, theta, L, S)
GTS = SEV = None
if gt is not None:
    try:
        GTS, SEV, SHARE = C.get_gt(key, gt_key, L, gt)
    except Exception as ex:  # noqa: BLE001
        st.sidebar.error(f"GT rules could not be applied: {ex}")
        GTS = None
n_sig = len(L.sigs)

tabs = st.tabs(["1 · Overview", "2 · Hierarchical merge", "3 · Infomap", "4 · Compare", "5 · Export"])

# ================================================================ overview
with tabs[0]:
    c = st.columns(5)
    c[0].metric("events", f"{len(L.ev_act):,}")
    c[1].metric("objects", f"{len(L.otype):,}")
    c[2].metric("object types", len(set(L.otype.values())))
    c[3].metric("signatures", n_sig)
    c[4].metric("edges", f"{len(S.edges)}", help=f"{S.n_all_edges} before the noise filter")
    if L.ev_sig.count(None):
        st.caption(f"{L.ev_sig.count(None):,} events have no objects: they get no scope.")
    if L.n_bad_time:
        st.warning(f"{L.n_bad_time:,} events have unreadable timestamps and are left out of the timelines.")
    st.markdown("#### Signals on this log")
    st.markdown("Look here before choosing weights: a signal that is constant or saturated cannot tell edges apart.")
    ov, corr, warns = C.signal_overview(S, B)
    for wtxt in warns:
        st.warning(wtxt, icon=":material/warning:")
    st.dataframe(ov.drop(columns="key").style.format({"share of hand-overs": "{:.0%}", "min": "{:.2f}", "max": "{:.2f}",
                                                       "mean (hand-over weighted)": "{:.2f}", "spread (sd)": "{:.2f}"}),
                 hide_index=True, width="stretch")
    l, r = st.columns([0.4, 0.6], gap="large")
    with l:
        colour = st.radio("Colour edges by", E.SIGNALS + ["b"], format_func=lambda k: V.SIGNAL_LABEL[k], key="ov_col")
    with r:
        vals = B.b if colour == "b" else {e: S.raw[e][colour] for e in S.edges}
        st.plotly_chart(V.graph(L, S, POS, edge_value=vals, value_name=V.SIGNAL_LABEL[colour], height=560,
                                title=f"{V.SIGNAL_LABEL[colour]} per edge"), width="stretch")
    with st.expander("All edges"):
        st.dataframe(E.edge_table(L, S, B).round(3), hide_index=True, width="stretch")
    with st.expander("All signatures"):
        st.dataframe(pd.DataFrame([{"signature": s, "events": L.n_events[s],
                                    "activities": ", ".join(f"{a} ({n})" for a, n in L.sig_acts[s].most_common(5))}
                                   for s in sorted(L.sigs, key=lambda s: -L.n_events[s])]), hide_index=True, width="stretch")

# ================================================================ merge
with tabs[1]:
    nmax = len(M.merges)
    kmin = n_sig - nmax
    c1, c2 = st.columns([0.3, 0.7])
    by = c1.radio("Choose the level by", ["number of scopes k", "merge height τ"], key="m_by")
    if by.startswith("number"):
        default_k = min(max(kmin, 4), n_sig)
        k = c2.slider("k", kmin, n_sig, default_k, key="m_k")
        n_m = M.n_merges_for(k=k); level = f"k{k}"
    else:
        hmax = max([m[0] for m in M.merges] + [0.01])
        tau = c2.slider("τ", 0.0, float(round(hmax + 0.01, 3)), float(round(V.cut_height(M, M.n_merges_for(k=min(4, n_sig))), 3)),
                        0.001, format="%.3f", key="m_tau")
        n_m = M.n_merges_for(tau=tau); level = f"tau{tau:.3f}"
    if kmin > 1:
        st.caption(f"The graph has {kmin} unconnected parts, so the merge cannot go below k = {kmin}.")
    parts_m = E.by_size(L, M.cut(n_m))
    smap_m = E.scope_map(parts_m)
    st.session_state["merge_result"] = (parts_m, level)
    l, r = st.columns([0.5, 0.5], gap="large")
    with l:
        st.plotly_chart(V.dendrogram(L, M, cut_x=V.cut_height(M, n_m) if 0 < n_m < nmax else None, ties=M.ties),
                        width="stretch")
        if M.ties:
            st.caption(f"⚖️ Ties at merge steps {', '.join(map(str, M.ties))}: equal distances are broken by key order (signature name).")
    with r:
        st.plotly_chart(V.graph(L, S, POS, edge_value=B.b_merge, scope_of=smap_m, height=560,
                                value_name="b'" if lam else "b", title=f"{len(parts_m)} scopes"), width="stretch")
    st.dataframe(C.scope_table(L, parts_m), hide_index=True, width="stretch")
    with st.expander("Merge order"):
        rows = []
        for st_, (d, p, q, kk) in enumerate(M.merges, 1):
            rows.append({"step": st_, "distance": round(d, 3), "joined": " + ".join("; ".join(M.leaves(x)) for x in (p, q)),
                         "groups left": n_sig - st_, "tie": "yes" if st_ in M.ties else ""})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if GTS:
        st.markdown("#### Against the ground truth")
        fm, mm = E.score(parts_m, GTS, SEV)
        a, b_ = st.columns([0.45, 0.55])
        a.metric("macro F1", f"{mm:.3f}")
        a.dataframe(fm.round(3), hide_index=True, width="stretch")
        curve = []
        for kk in range(n_sig, kmin - 1, -1):
            curve.append({"k": kk, "macro F1": E.score(M.cut(M.n_merges_for(k=kk)), GTS, SEV)[1]})
        b_.plotly_chart(V.f1_curve(curve, len(GTS)), width="stretch")

# ================================================================ infomap
with tabs[2]:
    c1, c2 = st.columns([0.3, 0.7])
    weight = c1.radio("Link weight", ["together", "steps"], key="im_w",
                      help="together = hand-overs × (1 − b); steps = hand-overs only")
    res = {mt: C.get_infomap(key, theta, WT, weight, mt, int(seed), int(trials), L, S, B_plain) for mt in mts}
    mt = c2.select_slider("Markov time", options=mts, value=0.75 if 0.75 in mts else mts[0], key="im_mt", help=C.tip("markov time"))
    parts_i = res[mt][0]
    smap_i = E.scope_map(parts_i)
    st.session_state["infomap_result"] = (parts_i, f"mt{mt:g}")
    l, r = st.columns([0.5, 0.5], gap="large")
    with l:
        st.plotly_chart(V.alluvial(L, {m: p for m, (p, _) in res.items()}, chosen=mt), width="stretch")
        rows = [{"Markov time": m, "modules": len(p),
                 **({"macro F1": round(E.score(p, GTS, SEV)[1], 3)} if GTS else {})} for m, (p, cl) in sorted(res.items())]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with r:
        st.plotly_chart(V.graph(L, S, POS, edge_value=B_plain.b, scope_of=smap_i, height=560,
                                title=f"Markov time {mt:g}: {len(parts_i)} modules"), width="stretch")
    st.dataframe(C.scope_table(L, parts_i), hide_index=True, width="stretch")
    if lam:
        st.caption("The distance balancer applies to the merge only; Infomap uses the plain b.")
    if GTS:
        fi, mi = E.score(parts_i, GTS, SEV)
        st.metric("macro F1", f"{mi:.3f}")
        st.dataframe(fi.round(3), hide_index=True, width="stretch")

# ================================================================ compare
with tabs[3]:
    pm, lv_m = st.session_state["merge_result"]
    pi, lv_i = st.session_state["infomap_result"]
    sm, si = E.scope_map(pm), E.scope_map(pi)
    st.markdown(f"Merge at **{lv_m}** ({len(pm)} scopes) vs Infomap at **{lv_i}** ({len(pi)} modules). "
                "Change the levels in tabs 2 and 3.")
    m = st.columns(3)
    m[0].metric("agreement (ARI, event-weighted)", f"{C.event_ari(L, sm, si):.2f}")
    if GTS:
        m[1].metric("merge macro F1", f"{E.score(pm, GTS, SEV)[1]:.3f}")
        m[2].metric("Infomap macro F1", f"{E.score(pi, GTS, SEV)[1]:.3f}")
    c = st.columns(2)
    c[0].plotly_chart(V.graph(L, S, POS, scope_of=sm, title="Hierarchical merge", height=480), width="stretch")
    c[1].plotly_chart(V.graph(L, S, POS, scope_of=si, title="Infomap", height=480), width="stretch")
    # signature pairs placed differently
    diff = []
    sigs = L.sigs
    for a_i, a in enumerate(sigs):
        for b2 in sigs[a_i + 1:]:
            tm, ti = sm[a] == sm[b2], si[a] == si[b2]
            if tm != ti and ((a, b2) in S.edges or (b2, a) in S.edges):
                diff.append({"pair": f"{a} — {b2}", "merge": "together" if tm else "apart",
                             "Infomap": "together" if ti else "apart",
                             "hand-overs": S.edges.get((a, b2), 0) + S.edges.get((b2, a), 0)})
    st.markdown("#### Connected signature pairs the methods place differently")
    if diff:
        st.dataframe(pd.DataFrame(diff).sort_values("hand-overs", ascending=False), hide_index=True, width="stretch")
    else:
        st.success("Both methods group every connected pair the same way.")

# ================================================================ export
with tabs[4]:
    meth = st.radio("Scopes from", ["Hierarchical merge", "Infomap"], horizontal=True, key="ex_meth")
    parts, lvl = st.session_state["merge_result"] if meth.startswith("Hier") else st.session_state["infomap_result"]
    st.markdown(f"Level **{lvl}** → **{len(parts)} scopes** (set in tab {'2' if meth.startswith('Hier') else '3'}).")
    procs, bad = E.make_rules(L, E.scope_map(parts))
    if bad:
        st.error(f"Rule check failed for {len(bad)} signatures: " + "; ".join(f"{s} → {h}" for s, _, h in bad))
    else:
        st.success(f"Rule check passed: all {len(E.scope_map(parts))} signatures match exactly their own scope "
                   f"under Procellar semantics ({sum(len(p['rules']) for p in procs)} rules).", icon=":material/verified:")
    src_tag = "merge" if meth.startswith("Hier") else "infomap"
    params = {"theta": theta, "weights": w, "balancer": lam if src_tag == "merge" else 0.0, "weight": weight,
              "seed": int(seed), "trials": int(trials)}
    f1 = E.score(parts, GTS, SEV) if GTS else None
    Bx = B if src_tag == "merge" else B_plain
    rep = C.report_md(log_name, L, S, Bx, params, meth, lvl, parts, bad, f1=f1, ties=M.ties)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ Procellar rules (JSON)", json.dumps(procs, indent=2), f"rules_{log_name}_{src_tag}_{lvl}.json",
                       "application/json", width="stretch", type="primary")
    c2.download_button("⬇ Report (Markdown)", rep, f"report_{log_name}_{src_tag}_{lvl}.md", "text/markdown",
                       width="stretch")
    st.link_button("Procellar on GitHub (import the rules there)", C.PROCELLAR_REPO, icon=":material/open_in_new:")
    t1, t2 = st.tabs(["Report preview", "Rules preview"])
    with t1:
        st.markdown(rep.replace("\n### ", "\n###### ").replace("\n## ", "\n##### ").replace("# Scope proposal", "#### Scope proposal", 1))
    with t2:
        st.json(procs, expanded=1)
