"""Guided explainer on the bundled Logistics log, with a free-explore toggle on three steps."""
import math

import pandas as pd
import streamlit as st

import common as C
import engine as E
import viz as V

D = E.DEFAULTS
key, L, G, GT = C.demo_bundle()
S = C.get_signals(key, 0.0, L, G)
POS = C.get_layout(key, 0.0, L, S)
GTS, SEV, SHARE = C.get_gt(key, "demo-gt", L, GT)
MAJ = C.gt_majority(SHARE)
GT_NAMES = sorted(GTS)
GT_SHORT = {g: g.replace(" Management", "").replace("Transportation", "Transport") for g in GT_NAMES}
GT_IDX = {g: i for i, g in enumerate(GT_NAMES, 1)}
GT_K = len(GTS)
B0, M0 = C.get_scores_merge(key, 0.0, C.wtuple(D["weights"]), 0.0, L, S)

FK, FKV = "{Container, Forklift}", "{Container, Forklift, Vehicle}"
LT, DT = "{Container, Handling Unit, Truck}", "{Container, Truck}"


def act(s, n=2):
    return ", ".join(a for a, _ in L.sig_acts[s].most_common(n))


def nm(s):
    return f"**{s}** ({act(s)})"


def gt_of(s):
    g, sh = MAJ.get(s, ("–", 0))
    return GT_SHORT.get(g, g)


def gtmap():
    return {s: GT_IDX[MAJ[s][0]] for s in L.sigs if s in MAJ}


def explore_controls(prefix, lam=False):
    st.caption("Free explore: these settings only apply inside this step.")
    w = {}
    cols = st.columns(4)
    for c, k in zip(cols, E.SIGNALS):
        w[k] = c.slider(V.SIGNAL_LABEL[k], 0.0, 2.0, float(D["weights"][k]), 0.1, key=f"{prefix}_w_{k}", help=C.tip(k))
    if sum(w.values()) == 0:
        st.warning("All weights are 0; using equal weights.")
        w = dict(D["weights"])
    b = 0.0
    if lam:
        b = st.slider("Distance balancer λ", 0.0, 1.0, 0.0, 0.1, key=f"{prefix}_lam", help=C.tip("balancer"))
    return w, b


# ======================================================================= steps
def step_intro():
    st.markdown(
        "An event log of a business rarely says where one process stops and the next starts. In an "
        "**object-centric** log (OCEL 2.0) this is even harder: there is no case id, only events and the objects "
        "they touch, and one container, truck or order runs through several pieces of work.")
    st.markdown(
        "A **process scope** a stretch of work where not strong change happens inside. Objects may cross scopes, but should do so cleanly. This tool looks "
        "for the places where it *does* change: other kinds of objects taking part, a change in batching, or a wait "
        "between two sections of work.")

    c = st.columns(4)
    c[0].metric("events", f"{len(L.ev_act):,}")
    c[1].metric("object types", len(set(L.otype.values())))
    c[2].metric("objects", f"{len(L.otype):,}")
    c[3].metric("activities", len(set(L.ev_act)))
    st.markdown(
        f"The log used here is the public [Logistics]({C.LOGISTICS_URL}) OCEL: customer orders, transport documents, goods "
        f"collected in handling units, trucks bringing containers to a terminal, forklifts and vehicles for export. "
        f"Its documentation describes **{GT_K} phases** ({', '.join(GT_SHORT[g] for g in GT_NAMES)}), which we use at "
        "the end to see how close each algorithm gets.")
    st.info("The walkthrough has 8 steps. Steps 4, 5, 6 and 7 have a **free explore** switch to change the settings yourself.",
            icon=":material/info:")


def step_signatures():
    l, r = st.columns([0.42, 0.58], gap="large")
    with l:
        st.markdown(
            "Every event touches a set of object types. That set is the event's **signature**. *Load Truck* events touch a "
            "container, a handling unit and a truck: signature {Container, Handling Unit, Truck}.")
        st.markdown(
            f"This log has **{len(L.n_events)} signatures**. They are the building blocks: every scope the tool proposes "
            "is a group of signatures, so every event of a signature ends up in the same scope. That keeps the output "
            "writable as Procellar rules on object types.")
        st.markdown("Bigger circle = more events.")
        df = pd.DataFrame([{"signature": s, "events": L.n_events[s], "activities": act(s, 3)}
                           for s in sorted(L.sigs, key=lambda s: -L.n_events[s])])
        st.dataframe(df, hide_index=True, width="stretch", height=300)
    with r:
        st.plotly_chart(V.graph(L, S, POS, title="Signatures (nodes)", height=600), width="stretch")


def step_handovers():
    l, r = st.columns([0.42, 0.58], gap="large")
    with l:
        st.markdown(
            "Follow one object through time: a handling unit is first collected (*Collect Goods*), then loaded onto a truck "
            "(*Load Truck*). Each step from one event of an object to its next event is a **hand-over**. When the two "
            "events have different signatures, the hand-over is an **edge** of the graph.")
        st.markdown("A hand-over is counted once per pair of events, even when several objects make the same step. "
                    "Thicker arrow = more hand-overs.")
        top = sorted(S.edges.items(), key=lambda kv: -kv[1])
        opts = [f"{x} → {y}" for (x, y), _ in top]
        pick = st.selectbox("Look at an edge", opts, index=0)
        e = top[opts.index(pick)][0]
        st.metric("hand-overs on this edge", f"{S.edges[e]:,}")
        pa = G.pair_acts.get(e, {})
        st.dataframe(pd.DataFrame([{"from activity": a, "to activity": b, "hand-overs": n} for (a, b), n in pa.most_common(6)]),
                     hide_index=True, width="stretch")
        st.caption(f"{len(S.edges)} edges in total. Hand-overs inside one signature are not edges; they only set the pace "
                   "of that signature (used in step 4).")
    with r:
        st.plotly_chart(V.graph(L, S, POS, highlight={e}, title="Hand-overs between signatures (edges)", height=600),
                        width="stretch")


def _examples():
    ex = {}
    for k in E.SIGNALS:
        cands = [(S.raw[e][k] * math.log1p(c), e) for e, c in S.edges.items() if S.raw[e][k] > 0]
        if cands:
            ex[k] = max(cands)[1]
    return ex


def _signal_text(k, e):
    x, y = e
    r, w = S.raw[e], S.why[e]
    if k == "participants":
        return (f"{x} and {y} share {w['participants']}. Jaccard distance = 1 − shared / all types = "
                f"**{r[k]:.2f}**. A new kind of object joining or leaving the work is a hint of a new piece of work.")
    if k == "multiplicity":
        return (f"For shared type **{w['multiplicity']}**, the number of objects per event changes at the typical hand-over: "
                f"**{r[k]:.2f}** (0 = same number, near 1 = many go into one event, e.g. containers loaded on one vessel).")
    if k == "wait":
        t = w["wait"]
        ratio = t.split(" x")[-1] if " x" in t else "–"
        return (f"For **{t.split(' x')[0]}**, the median gap at this hand-over is **{ratio}×** the slower working pace on "
                f"either side. Signal = 1 − 1/ratio = **{r[k]:.2f}**. A wait that belongs to neither side points at a border.")
    return (f"The inside pace of **{w['pace_change']}** differs between the two sides: **{r[k]:.2f}** "
            "(relative difference). Catches slow → fast work without a pause. Off by default.")


NOTATION = ("For an edge x → y: $T_x$ = object types of signature x, $S = T_x \\cap T_y$ = shared types, "
            "$H_{xy}$ = hand-overs from x to y (pairs of events $(a, b)$), $n_T(e)$ = number of objects of type T in "
            "event e, $\\mathrm{pace}_T(x)$ = median time between two consecutive events of one T object inside x.")
FORMULA = {
    "participants": r"p(x\to y)=1-\frac{|T_x\cap T_y|}{|T_x\cup T_y|}",
    "multiplicity": r"m(x\to y)=\max_{T\in S}\ \operatorname*{median}_{(a,b)\in H_{xy}}"
                    r"\left(1-\frac{\min(n_T(a),\,n_T(b))}{\max(n_T(a),\,n_T(b))}\right)",
    "wait": r"t(x\to y)=\max_{T\in S}\ \max\!\left(0,\;1-\frac{1}{r_T}\right),\qquad "
            r"r_T=\frac{\operatorname{median}_{H_{xy}}\,\mathrm{gap}_T}{\max(\mathrm{pace}_T(x),\,\mathrm{pace}_T(y))}",
    "pace_change": r"c(x\to y)=\max_{T\in S}\ \frac{|\mathrm{pace}_T(x)-\mathrm{pace}_T(y)|}{\max(\mathrm{pace}_T(x),\,\mathrm{pace}_T(y))}",
}
WHERE = {
    "participants": "T_x, T_y = object types of the two signatures. 0 = same types, close to 1 = almost nothing shared.",
    "multiplicity": "S = shared types, H_xy = hand-overs x → y, n_T(e) = objects of type T in event e. "
                    "0 = same number of objects, close to 1 = many objects go into one event (or the reverse).",
    "wait": "gap_T = time between the two events of a T object at the hand-over; pace_T = median time between two "
            "consecutive events of a T object inside one signature (one side missing → the other side). "
            "0 = the hand-over is not slower than the work on either side.",
    "pace_change": "pace_T = median time between two consecutive events of a T object inside the signature; "
                   "both sides needed. 0 = same rhythm, close to 1 = very different rhythm.",
}


def step_signals():
    fx = st.toggle("Free explore", key="fx_signals")
    ex = _examples()
    l, r = st.columns([0.42, 0.58], gap="large")
    with l:
        st.markdown(
            "Each edge x → y gets four signals between 0 and 1. High means *the nature of the work changes here*:")
        st.markdown(
            "- **Participants change**: other kinds of objects take part\n"
            "- **Batching change**: the number of objects per event changes\n"
            "- **Waiting at the hand-over**: objects wait between the two pieces of work\n"
            "- **Change of pace**: the rhythm of work changes (off by default)")
        if fx:
            colour = st.radio("Colour edges by", E.SIGNALS, format_func=lambda k: V.SIGNAL_LABEL[k], horizontal=True, key="fx_sig_col")
            opts = sorted(S.edges, key=lambda e: -S.edges[e])
            e = st.selectbox("Edge", opts, format_func=lambda e: f"{e[0]} → {e[1]}  ({S.edges[e]:,})", key="fx_sig_edge")
        else:
            colour = st.radio("Example", [k for k in E.SIGNALS if k in ex], format_func=lambda k: V.SIGNAL_LABEL[k],
                              horizontal=True, key="g_sig")
            e = ex[colour]
        st.markdown(f"##### {V.SIGNAL_LABEL[colour]}")
        st.latex(FORMULA[colour])
        st.caption(WHERE[colour])
        st.markdown(f"**{e[0]} → {e[1]}** · {S.edges[e]:,} hand-overs")
        st.markdown(_signal_text(colour, e))
        st.plotly_chart(V.signal_bars(S.raw[e]), width="stretch")
        with st.expander("All four formulas"):
            st.markdown(NOTATION)
            for k in E.SIGNALS:
                st.markdown(f"**{V.SIGNAL_LABEL[k]}**")
                st.latex(FORMULA[k])
    with r:
        vals = {ed: S.raw[ed][colour] for ed in S.edges}
        st.plotly_chart(V.graph(L, S, POS, edge_value=vals, value_name=V.SIGNAL_LABEL[colour], highlight={e},
                                title=f"{V.SIGNAL_LABEL[colour]} on every edge", height=600), width="stretch")
        fire = sum(1 for v in vals.values() if v > 0)
        st.caption(f"{V.SIGNAL_LABEL[colour]} is above 0 on {fire} of {len(vals)} edges.")


def step_b():
    fx = st.toggle("Free explore", key="fx_b")
    l, r = st.columns([0.42, 0.58], gap="large")
    with l:
        st.markdown(
            "The four signals are combined into one **boundary score b** per edge: their weighted mean. "
            "With the thesis settings, participants, batching and waiting each count once and change of pace is off.")
        st.latex(r"b(x\to y)=\frac{\sum_s w_s\,\mathrm{signal}_s(x\to y)}{\sum_s w_s}")
        if fx:
            w, _ = explore_controls("fxb")
        else:
            w = dict(D["weights"])
            st.markdown("**Weights:** " + ", ".join(f"{V.SIGNAL_LABEL[k]} {w[k]:g}" for k in E.SIGNALS))
        B = E.scores(S, w, 0.0)
        tab = E.edge_table(L, S, B)[["from", "to", "steps", "participants", "multiplicity", "wait", "pace_change", "b"]]
        st.dataframe(tab.sort_values("b").round(3), hide_index=True, width="stretch", height=280)
        st.markdown(
            "Low b = the two signatures look like the same piece of work. The weights are the analyst's choice: they say "
            "what kind of change you care about. The tool's overview page helps decide them for your log.")
    with r:
        st.plotly_chart(V.graph(L, S, POS, edge_value=B.b, show_edge_labels=True, title="Boundary score b per edge",
                                height=580), width="stretch")


def step_merge():
    fx = st.toggle("Free explore", key="fx_merge")
    if fx:
        with st.container(border=True):
            w, lam = explore_controls("fxm", lam=True)
        B, M = C.get_scores_merge(key, 0.0, C.wtuple(w), lam, L, S)
    else:
        B, M = B0, M0
    n = len(M.merges)
    st.markdown(
        "**Hierarchical merging** starts with every signature as its own group. At every step it joins the two connected "
        "groups with the smallest **merge distance**: the mean b over all hand-overs between them (both directions). The result is a tree that you can cut at the number of scopes you want.")
    st_ = st.slider("Merge step", 0, n, 1 if not fx else 0, key="m_step" + ("_fx" if fx else ""),
                    help="0 = nothing merged yet")
    parts = E.by_size(L, M.cut(st_))
    smap = E.scope_map(parts)
    l, r = st.columns([0.5, 0.5], gap="large")
    with l:
        if st_ == 0:
            st.markdown(f"**{len(L.sigs)} groups**, one per signature. Move the slider to start merging.")
            hl = set()
        else:
            d, p, q, k = M.merges[st_ - 1]
            A, Bg = M.leaves(p), M.leaves(q)
            between = [(e, c) for e, c in S.edges.items() if (e[0] in A and e[1] in Bg) or (e[0] in Bg and e[1] in A)]
            hl = {e for e, _ in between}
            st.markdown(f"**Step {st_}: merge distance {d:.3f}** → {len(parts)} groups left")
            st.markdown("Joined: " + " + ".join("{" + "; ".join(g) + "}" if len(g) > 1 else g[0] for g in (A, Bg)))
            st.dataframe(pd.DataFrame([{"edge": f"{e[0]} → {e[1]}", "hand-overs": c, "b": round(B.b_merge[e], 3)}
                                       for e, c in sorted(between, key=lambda t: -t[1])]), hide_index=True, width="stretch")
            if st_ in M.ties:
                st.warning("**Tie**: another pair had exactly the same distance. Ties are broken by key order "
                           "(signature name order), not by a signal.", icon=":material/balance:")
            ga, gb = {gt_of(s) for s in A}, {gt_of(s) for s in Bg}
            if ga != gb or len(ga) > 1:
                st.error(f"Crosses a documented phase border: {', '.join(sorted(ga))} + {', '.join(sorted(gb))}.",
                         icon=":material/call_split:")
            else:
                st.success(f"Both sides belong to the same documented phase ({', '.join(ga)}).", icon=":material/check:")
            if not fx and st_ == 1 and {FK, FKV} == set(A) | set(Bg):
                st.markdown(
                    f"👉 The **first merge** joins the forklift work ({act(FK, 3)}) with *Load to Vehicle*. "
                    "All signals are low: the same container and forklift, one container per event, no wait. "
                    "But the documentation puts them in different phases (Transportation vs Export). "
                    "Locally this is one piece of work; only the documented process model draws a border here.")
        st.plotly_chart(V.dendrogram(L, M, cut_x=(V.cut_height(M, st_) if 0 < st_ < n else None), upto=st_, ties=M.ties),
                        width="stretch")
    with r:
        st.plotly_chart(V.graph(L, S, POS, edge_value=B.b_merge, scope_of=smap, highlight=hl,
                                title=f"Groups after step {st_}", height=600), width="stretch")
    if fx:
        st.markdown("##### How close is each level to the documented phases?")
        rows = []
        for kk in range(len(L.sigs), 0, -1):
            nm_ = M.n_merges_for(k=kk)
            if nm_ > len(M.merges):
                continue
            g = M.cut(nm_)
            rows.append({"k": len(g), "macro F1": E.score(g, GTS, SEV)[1]})
        st.plotly_chart(V.f1_curve(rows, GT_K), width="stretch")
        st.caption("Macro F1: for every documented phase the best-matching scope, then the mean. k = 12 means no merging.")


def step_infomap():
    fx = st.toggle("Free explore", key="fx_im")
    st.markdown(
        "**Infomap** looks at the graph as a flow. Imagine a walker following objects from event to event. A good scope "
        "is a group of signatures the walker rarely leaves. Infomap finds the grouping that describes the walk with the "
        "shortest code (the *map equation*). Edges are weighted by **hand-overs × (1 − b)**: many hand-overs that stay "
        "in the same kind of work hold signatures together.")
    mts = list(D["markov_times"])
    weight, trials = "together", D["trials"]
    if fx:
        with st.container(border=True):
            c1, c2, c3 = st.columns(3)
            weight = c1.radio("Link weight", ["together", "steps"], horizontal=True,
                              help="together = hand-overs × (1 − b); steps = hand-overs only (ignores the signals)")
            extra = c2.number_input("Add a Markov time", 0.05, 100.0, 0.35, 0.05)
            trials = c3.slider("Trials", 1, 50, D["trials"])
            if extra not in mts:
                mts = sorted(mts + [extra])
    res = {mt: C.get_infomap(key, 0.0, C.wtuple(D["weights"]), weight, mt, D["seed"], trials, L, S, B0)[0] for mt in mts}
    mt = st.select_slider("Markov time (zoom level)", options=mts, value=0.75 if 0.75 in mts else mts[0],
                          help=C.tip("markov time"), key="im_mt" + ("_fx" if fx else ""))
    parts = res[mt]
    smap = E.scope_map(parts)
    l, r = st.columns([0.5, 0.5], gap="large")
    with l:
        st.markdown(f"**Markov time {mt}: {len(parts)} modules.** Small Markov time = the walker is watched often → "
                    "many small modules; large = few big ones. Unlike the merge tree, modules at different Markov times "
                    "are **not nested**: signatures can switch between modules.")
        st.plotly_chart(V.alluvial(L, res, chosen=mt), width="stretch")
        st.caption("Each band is a signature (height = events), coloured by its module at the chosen Markov time.")
        if not fx and mt == 0.75:
            st.markdown(f"👉 At 0.75, *Drive to Terminal* {DT} sits with *Load Truck*: the truck goes back and forth "
                        "between the two all day, so the walker keeps circling there. The merge separates them, because "
                        "the truck's return trip has a high wait signal.")
    with r:
        st.plotly_chart(V.graph(L, S, POS, edge_value=B0.b, scope_of=smap, title=f"Infomap modules at Markov time {mt}",
                                height=600), width="stretch")


def _aligned(parts):
    """Number each found scope after the GT phase it shares most events with (greedy, one scope per phase),
    so colours match the GT view. Unmatched scopes get numbers after the GT phases."""
    ev = [set().union(*(SEV[s] for s in g)) for g in parts]
    pairs = sorted(((len(ev[i] & GTS[g]), i, g) for i in range(len(parts)) for g in GT_NAMES), reverse=True)
    lab, used = {}, set()
    for n, i, g in pairs:
        if n and i not in lab and g not in used:
            lab[i] = GT_IDX[g]; used.add(g)
    nxt = len(GT_NAMES) + 1
    for i in range(len(parts)):
        if i not in lab:
            lab[i] = nxt; nxt += 1
    smap = {s: lab[i] for i, g in enumerate(parts) for s in g}
    names = {v: (f"≈ {GT_SHORT[GT_NAMES[v - 1]]}" if v <= len(GT_NAMES) else f"extra {v - len(GT_NAMES)}")
             for v in lab.values()}
    return smap, names


def step_compare():
    st.markdown(f"Both methods at **{GT_K} scopes** (the number of documented phases) and the phases themselves, on the "
                "same graph. Each scope gets the colour of the phase it overlaps most, so switching views shows "
                "where the methods differ from the documentation.")
    pm = E.by_size(L, M0.cut(M0.n_merges_for(k=GT_K)))
    pi = C.get_infomap(key, 0.0, C.wtuple(D["weights"]), "together", 0.75, D["seed"], D["trials"], L, S, B0)[0]
    am, nm_m = _aligned(pm)
    ai, nm_i = _aligned(pi)
    gtm = gtmap()
    gt_names = {v: GT_SHORT[k] for k, v in GT_IDX.items()}
    view = st.segmented_control("Show", ["Merge, k = 4", "Infomap, Markov time 0.75", "Documented phases (GT)"],
                                default="Merge, k = 4", key="cmp_view")
    view = view or "Merge, k = 4"
    l, r = st.columns([0.45, 0.55], gap="medium")
    with r:
        if view.startswith("Merge"):
            fig = V.graph(L, S, POS, scope_of=am, scope_names=nm_m, title="Hierarchical merge, k = 4", height=600)
        elif view.startswith("Infomap"):
            fig = V.graph(L, S, POS, scope_of=ai, scope_names=nm_i, title="Infomap, Markov time 0.75", height=600)
        else:
            fig = V.graph(L, S, POS, scope_of=gtm, scope_names=gt_names, title="Documented phases (majority per signature)", height=600)
        st.plotly_chart(fig, width="stretch")
    fm, mm = E.score(pm, GTS, SEV)
    fi, mi = E.score(pi, GTS, SEV)
    with l:
        rows = []
        for s in sorted(L.sigs, key=lambda s: (gtm.get(s, 99), -L.n_events[s])):
            rows.append({"signature": s.strip("{}"), "GT": gt_names.get(gtm.get(s), "–"),
                         "merge": nm_m[am[s]].replace("≈ ", ""), "Infomap": nm_i[ai[s]].replace("≈ ", "")})
        df = pd.DataFrame(rows)

        def mark(row):
            return ["", "", *("background-color: #fde8e8" if row[c] != row["GT"] else "" for c in ("merge", "Infomap"))]
        st.dataframe(df.style.apply(mark, axis=1), hide_index=True, width="stretch", height=455,
                     column_config={"signature": st.column_config.TextColumn(width="medium"),
                                    "GT": st.column_config.TextColumn(width="small"),
                                    "merge": st.column_config.TextColumn(width="small"),
                                    "Infomap": st.column_config.TextColumn(width="small")})
        st.caption("Red = placed with a different phase than the documentation.")
    t = pd.DataFrame({"GT phase": [GT_SHORT[g] for g in fm["GT scope"]], "merge F1": fm["F1"].round(2), "Infomap F1": fi["F1"].round(2)})
    l2, r2 = st.columns([0.38, 0.62], gap="large")
    with l2:
        st.dataframe(t, hide_index=True, width="stretch")
        m = st.columns(3)
        m[0].metric("merge F1", f"{mm:.2f}")
        m[1].metric("Infomap F1", f"{mi:.2f}")
        m[2].metric("ARI", f"{C.event_ari(L, E.scope_map(pm), E.scope_map(pi)):.2f}",
                    help="Adjusted Rand index between the two scope sets, weighted by events")
    with r2:
        st.markdown(
            "**What to read from this**\n"
            "- Both join *Collect Goods* with *Load Truck* through the handling unit, and both join the forklift work with "
            "*Load to Vehicle* (the step-1 merge). The phases cut through busy object flow there.\n"
            "- The main difference is *Drive to Terminal*: Infomap keeps it with *Load Truck* (the truck loop, as in the GT), "
            "the merge puts it with the forklift block.\n"
            "- Merge cuts where work waits. Infomap keeps loops but glues busy hand-offs.\n"
            "- Leaving every signature on its own (k = 12) scores highest on this log: wrong early merges cost more than "
            "not merging. The documented phases are not the only reasonable reading of the log.")


def step_takeaways():
    st.markdown(
        "- **Signatures and hand-overs** turn an object-centric log into a graph that favors boundary searching.\n"
        "- **Four signals** say where the nature of work changes. Their weighted mean is the **boundary score b**.\n"
        "- **Hierarchical merging** builds a nested tree. The cut level can be chosen (k or a height). It follows local change.\n"
        "- **Infomap** follows the flow of objects. Zoom level can be chosen(Markov time). Loops hold together.\n"
        "- Neither method wins everywhere. Both are **proposals**: check the borders against process documentation, name the scopes, "
        "and write them down as Procellar rules.")
    st.page_link("views/workbench.py", label="Try it on your own log", icon=":material/arrow_forward:")


STEPS = [("Why scopes?", step_intro), ("Signatures", step_signatures), ("Hand-overs", step_handovers),
         ("Four signals", step_signals), ("Boundary score", step_b), ("Hierarchical merge", step_merge),
         ("Infomap", step_infomap), ("Compare", step_compare), ("Take-aways", step_takeaways)]
names = [n for n, _ in STEPS]

if "ex_step" not in st.session_state:
    st.session_state.ex_step = 0


def go(i):
    st.session_state.ex_step = max(0, min(len(STEPS) - 1, i))


st.title("How the two algorithms find borders")
pick = st.segmented_control("Step", [f"{i + 1}. {n}" for i, n in enumerate(names)],
                            default=f"{st.session_state.ex_step + 1}. {names[st.session_state.ex_step]}",
                            key=f"seg_{st.session_state.ex_step}", label_visibility="collapsed")
if pick:
    i = int(pick.split(".")[0]) - 1
    if i != st.session_state.ex_step:
        go(i); st.rerun()
i = st.session_state.ex_step
st.subheader(f"{i + 1}. {names[i]}")
STEPS[i][1]()
st.divider()
b1, _, b2 = st.columns([1, 4, 1])
if i > 0:
    b1.button("← Back", on_click=go, args=(i - 1,), width="stretch")
if i < len(STEPS) - 1:
    b2.button("Next →", on_click=go, args=(i + 1,), type="primary", width="stretch")
