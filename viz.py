"""Figures for the scope tool (Plotly). One fixed graph layout per log, reused in every view."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import plotly.graph_objects as go

SCOPE_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]
OTHER = "#9c9c9c"
TEXT, TEXT2, GRID = "#191919", "#666666", "#e6e6e6"
B_SCALE = [[0.0, "#e9e4df"], [0.35, "#f6b48a"], [0.7, "#e2621f"], [1.0, "#7a2e06"]]   # one hue, light -> dark
SIGNAL_LABEL = {"participants": "Participants change", "multiplicity": "Batching change",
                "wait": "Waiting at the hand-over", "pace_change": "Change of pace", "b": "Boundary score b"}


def scope_color(i):
    return SCOPE_COLORS[(i - 1)] if i and 1 <= i <= len(SCOPE_COLORS) else OTHER


def short(s, L=None):
    """Readable node label: signature + main activity."""
    return s


# ---------------------------------------------------------------- layout
def _flow_order(sigs, edges):
    """Eades-Lin-Smyth greedy: sinks to the back, sources to the front, else the node with the largest
    (outgoing - incoming) hand-overs. Ties by name."""
    left = set(sigs)
    front, back = [], []
    def out_w(v): return sum(c for (a, b), c in edges.items() if a == v and b in left and b != v)
    def in_w(v): return sum(c for (a, b), c in edges.items() if b == v and a in left and a != v)
    while left:
        changed = True
        while changed:
            changed = False
            for v in sorted(left):
                if out_w(v) == 0 and in_w(v) > 0:
                    back.insert(0, v); left.remove(v); changed = True
            for v in sorted(left):
                if in_w(v) == 0:
                    front.append(v); left.remove(v); changed = True
        if left:
            v = max(sorted(left), key=lambda v: out_w(v) - in_w(v))
            front.append(v); left.remove(v)
    order = front + back
    # sifting: move each node to the position with the least backward hand-over weight
    w = defaultdict(int)
    for (a, b), c in edges.items():
        if a != b:
            w[(a, b)] += c
    def back_w(o):
        pos = {v: i for i, v in enumerate(o)}
        return sum(c for (a, b), c in w.items() if pos[a] > pos[b])
    best = back_w(order)
    for _ in range(3):
        improved = False
        for v in list(order):
            rest = [u for u in order if u != v]
            cands = [(back_w(rest[:i] + [v] + rest[i:]), i) for i in range(len(rest) + 1)]
            bw, i = min(cands)
            if bw < best:
                order, best, improved = rest[:i] + [v] + rest[i:], bw, True
        if not improved:
            break
    return order


def layout(L, edges):
    """Layered left-to-right layout. Node order from a greedy feedback-arc-set ordering (Eades et al. 1993)
    on hand-over counts, cut into columns; rows inside a column ordered by neighbour barycentre.
    Deterministic: same log -> same positions."""
    sigs = L.sigs
    n = len(sigs)
    idx = {s: i for i, s in enumerate(sigs)}
    if n == 1:
        return {sigs[0]: (0.5, 0.5)}
    order = _flow_order(sigs, edges)
    per = max(2, round(math.sqrt(n / 2)))              # rows per column: 12 signatures -> 2 rows x 6 columns
    ncol = math.ceil(n / per)
    cols = [order[i:i + per] for i in range(0, n, per)]
    ncol = len(cols)
    nb = defaultdict(float)
    for (x, y), c in edges.items():
        nb[(x, y)] += math.log1p(c); nb[(y, x)] += math.log1p(c)
    ypos = {}
    for c in cols:
        for j, s in enumerate(c):
            ypos[s] = (j + 0.5) / len(c)
    for _ in range(6):                                   # barycentre sweeps, left->right and back
        for rng in (range(1, ncol), range(ncol - 2, -1, -1)):
            for ci in rng:
                def bary(s):
                    ws = [(ypos[t], wt) for (a, t), wt in nb.items() if a == s and t not in cols[ci]]
                    return sum(y * wt for y, wt in ws) / sum(wt for _, wt in ws) if ws else ypos[s]
                cols[ci] = sorted(cols[ci], key=lambda s: (bary(s), s))
                for j, s in enumerate(cols[ci]):
                    ypos[s] = (j + 0.5) / len(cols[ci])
    pos = {}
    for ci, c in enumerate(cols):
        x = ci / (ncol - 1) if ncol > 1 else 0.5
        shift = 0.12 if ci % 2 else -0.12                # stagger columns so edges do not run through nodes
        for j, s in enumerate(c):
            y = (j + 0.5) / len(c) if len(c) > 1 else 0.5 + shift
            pos[s] = (x, min(1.0, max(0.0, y + (shift if len(c) > 1 else 0))))
    return pos


# ---------------------------------------------------------------- signature graph
def graph(L, S, pos, *, edge_value=None, scope_of=None, highlight=None, title=None, height=520,
          show_edge_labels=False, value_name="b", node_text=None, dim_other=False, cmax=1.0, scope_names=None):
    """edge_value: (x, y) -> 0..1 (colour); scope_of: sig -> scope number (node colour);
    highlight: set of edges or signatures to emphasise."""
    highlight = set(highlight or [])
    fig = go.Figure()
    maxc = max(S.edges.values()) if S.edges else 1
    shapes_x, shapes_y, mids = [], [], []
    ann = []
    for (x, y), c in sorted(S.edges.items(), key=lambda kv: kv[1]):
        (x0, y0), (x1, y1) = pos[x], pos[y]
        two_way = (y, x) in S.edges
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy) or 1
        ox, oy = (-dy / ln * 0.012, dx / ln * 0.012) if two_way else (0, 0)
        # shorten so the arrow ends before the node
        sh = 0.035
        sx0, sy0 = x0 + dx / ln * sh + ox, y0 + dy / ln * sh + oy
        sx1, sy1 = x1 - dx / ln * sh + ox, y1 - dy / ln * sh + oy
        v = edge_value.get((x, y)) if edge_value else None
        col = _scale_color(v / cmax if cmax else 0) if v is not None else "#b8b8b8"
        width = 1.2 + 6 * math.sqrt(c / maxc)
        is_hl = (x, y) in highlight
        op = 1.0 if (not highlight or is_hl or not dim_other) else 0.25
        ann.append(dict(x=sx1, y=sy1, ax=sx0, ay=sy0, xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=2, arrowsize=0.9, arrowwidth=width if not is_hl else width + 1.5,
                        arrowcolor=col if not is_hl else "#191919", opacity=op, text=""))
        mids.append(((sx0 + sx1) / 2, (sy0 + sy1) / 2, x, y, c, v))
    fig.update_layout(annotations=ann)
    # edge hover targets
    if mids:
        hov = []
        for mx, my, x, y, c, v in mids:
            raw = S.raw[(x, y)]
            t = (f"<b>{x} → {y}</b><br>hand-overs: {c:,}<br>"
                 + "<br>".join(f"{SIGNAL_LABEL[k]}: {raw[k]:.2f}" for k in raw))
            if v is not None:
                t += f"<br><b>{value_name}: {v:.3f}</b>"
            hov.append(t)
        fig.add_trace(go.Scatter(x=[m[0] for m in mids], y=[m[1] for m in mids], mode="markers+text" if show_edge_labels else "markers",
                                 marker=dict(size=14, color="rgba(0,0,0,0)"),
                                 text=[f"{m[5]:.2f}" if m[5] is not None else "" for m in mids] if show_edge_labels else None,
                                 textfont=dict(size=10, color=TEXT2), textposition="middle center",
                                 hovertext=hov, hoverinfo="text", showlegend=False))
    # nodes
    sigs = L.sigs
    tot = max(L.n_events.values())
    cols, sizes, hov, lines = [], [], [], []
    for s in sigs:
        sc = scope_of.get(s) if scope_of else None
        cols.append(scope_color(sc) if scope_of else "#ffffff")
        sizes.append(14 + 26 * math.sqrt(L.n_events[s] / tot))
        acts = ", ".join(f"{a} ({n:,})" for a, n in L.sig_acts[s].most_common(4))
        hov.append(f"<b>{s}</b><br>{L.n_events[s]:,} events<br>{acts}" + (f"<br>{(scope_names or {}).get(sc, f'scope {sc}')}" if sc else ""))
        lines.append("#191919" if s in highlight else "#555555")
    fig.add_trace(go.Scatter(x=[pos[s][0] for s in sigs], y=[pos[s][1] for s in sigs], mode="markers",
                             marker=dict(size=sizes, color=cols, line=dict(width=[3 if s in highlight else 1.2 for s in sigs], color=lines)),
                             hovertext=hov, hoverinfo="text", showlegend=False))
    # labels: signature (bold) + main activity
    for s in sigs:
        main = node_text(s) if node_text else L.sig_acts[s].most_common(1)[0][0]
        r = 7 + 13 * math.sqrt(L.n_events[s] / tot)
        fig.add_annotation(x=pos[s][0], y=pos[s][1], text=f"<b>{_wrap(s.strip('{}'))}</b><br><span style='color:{TEXT2}'>{main}</span>",
                           showarrow=False, yshift=-(r + 4), yanchor="top", font=dict(size=10, color=TEXT), align="center",
                           bgcolor="rgba(255,255,255,0.85)", borderpad=1)
    # legend for scopes
    if scope_of:
        for sc in sorted(set(scope_of.values())):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=scope_color(sc)),
                                     name=(scope_names or {}).get(sc, f"scope {sc}")))
    if edge_value is not None:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(colorscale=B_SCALE, cmin=0, cmax=cmax, color=[0], showscale=True,
                                             colorbar=dict(title=dict(text=value_name, side="right"), thickness=10, len=0.6, x=1.02)),
                                 hoverinfo="skip", showlegend=False))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40 if title else 10, b=10),
                      title=dict(text=title, x=0, font=dict(size=14)) if title else None,
                      xaxis=dict(visible=False, range=[-0.13, 1.13]), yaxis=dict(visible=False, range=[-0.08, 1.0]),
                      plot_bgcolor="white", paper_bgcolor="white", hoverlabel=dict(bgcolor="white", font_size=12),
                      legend=dict(orientation="h", y=-0.02, x=0, font=dict(size=11)))
    return fig


def _wrap(text, width=18):
    """Break a type list over lines at the commas, about `width` characters per line."""
    out, line = [], ""
    for part in text.split(", "):
        cand = part if not line else line + ", " + part
        if line and len(cand) > width:
            out.append(line + ","); line = part
        else:
            line = cand
    out.append(line)
    return "<br>".join(out)


def _scale_color(v):
    v = max(0.0, min(1.0, float(v)))
    for (a, ca), (b, cb) in zip(B_SCALE, B_SCALE[1:]):
        if v <= b:
            f = (v - a) / (b - a) if b > a else 0
            return _mix(ca, cb, f)
    return B_SCALE[-1][1]


def _mix(c1, c2, f):
    h1 = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    h2 = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(a + (b - a) * f):02x}" for a, b in zip(h1, h2))


# ---------------------------------------------------------------- dendrogram
def dendrogram(L, M, cut_x=None, upto=None, height=None, ties=None):
    """Horizontal dendrogram as in the notebook visual. upto: only draw the first `upto` merges."""
    order = []

    def walk(k):
        n = M.node[k]
        if n[0] == "leaf":
            order.append(k); return
        for c in sorted((n[1], n[2]), key=lambda c: -len(M.leaves(c))):
            walk(c)
    for k in M.tops:
        walk(k)
    pos = {k: (0.0, y) for y, k in enumerate(order)}
    drawn = {d[3] for d in M.merges[:upto]} if upto is not None else {d[3] for d in M.merges}
    step_of = {m[3]: i for i, m in enumerate(M.merges, 1)}
    fig = go.Figure()
    xs, ys, lab_x, lab_y, lab_t, hov = [], [], [], [], [], []
    ghost_x, ghost_y = [], []

    def place(k):
        if k in pos:
            return pos[k]
        _, p, q, d = M.node[k]
        (xp, yp), (xq, yq) = place(p), place(q)
        tx, ty = (xs, ys) if k in drawn else (ghost_x, ghost_y)
        for x0, y0 in ((xp, yp), (xq, yq)):
            tx += [x0, d, None]; ty += [y0, y0, None]
        tx += [d, d, None]; ty += [yp, yq, None]
        if k in drawn:
            lab_x.append(d); lab_y.append((yp + yq) / 2)
            st = step_of[k]
            lab_t.append(f"{st}")
            hov.append(f"merge step {st}: d = {d:.3f}" + (" (tie)" if ties and st in ties else ""))
        pos[k] = (d, (yp + yq) / 2)
        return pos[k]
    for k in M.tops:
        place(k)
    fig.add_trace(go.Scatter(x=ghost_x, y=ghost_y, mode="lines", line=dict(color="#d9d9d9", width=1.5), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#257de0", width=2), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=lab_x, y=lab_y, mode="markers+text", marker=dict(size=16, color="white", line=dict(color="#257de0", width=1.5)),
                             text=lab_t, textfont=dict(size=9, color=TEXT), hovertext=hov, hoverinfo="text", showlegend=False))
    if cut_x is not None:
        fig.add_vline(x=cut_x, line=dict(color="#ec4647", dash="dash", width=1.5))
        fig.add_annotation(x=cut_x, y=-0.8, text=f"cut {cut_x:.3f}", showarrow=False, font=dict(color="#c0392b", size=10), xanchor="left")
    labels = [f"{M.node[k][1]}  ({L.n_events[M.node[k][1]]:,})" for k in order]
    xmax = max([m[0] for m in M.merges] + [0.1]) * 1.12
    fig.update_layout(height=height or (34 * len(order) + 90), margin=dict(l=10, r=20, t=10, b=40),
                      xaxis=dict(title="merge distance", range=[0, xmax],
                                 gridcolor=GRID, zeroline=False),
                      yaxis=dict(tickvals=list(range(len(order))), ticktext=labels, autorange="reversed", tickfont=dict(size=11)),
                      plot_bgcolor="white", paper_bgcolor="white")
    return fig


def cut_height(M, n_m):
    lo = M.merges[n_m - 1][0] if n_m > 0 else 0.0
    hi = M.merges[n_m][0] if n_m < len(M.merges) else lo + 0.05
    return (lo + hi) / 2


# ---------------------------------------------------------------- alluvial (Markov-time sweep)
def alluvial(L, results, chosen=None, height=520, min_h=0.04, gap=0.03):
    """Port of the notebook's alluvial cell: columns = Markov times, blocks = modules,
    bands = signatures (height = events), colour = module at the chosen Markov time."""
    mts = sorted(results)
    sigs = L.sigs
    tot = sum(L.n_events.values())
    h = {s: max(min_h, L.n_events[s] / tot) for s in sigs}
    orders, prev_rank = [], None
    for mt in mts:
        groups = [list(g) for g in results[mt]]
        if prev_rank is None:
            groups = [sorted(g, key=lambda s: -L.n_events[s]) for g in groups]
        else:
            groups = [sorted(g, key=lambda s: prev_rank[s]) for g in groups]
            groups.sort(key=lambda g: sum(prev_rank[s] for s in g) / len(g))
        order = [s for g in groups for s in g]
        orders.append(groups)
        prev_rank = {s: i for i, s in enumerate(order)}
    seg, blocks = [], []
    for groups in orders:
        y, pos, blk = 0.0, {}, []
        for g in groups:
            y0 = y
            for s in g:
                pos[s] = (y, y + h[s]); y += h[s]
            blk.append((y0, y, g)); y += gap
        seg.append(pos); blocks.append(blk)
    ref = mts.index(chosen) if chosen in mts else 0
    ref_mod = {s: k for k, g in enumerate(results[mts[ref]], 1) for s in g}
    W = 0.12
    shapes = []
    hx, hy, ht = [], [], []
    for c in range(len(mts) - 1):
        for s in sigs:
            (a0, a1), (b0, b1) = seg[c][s], seg[c + 1][s]
            x0, x1 = c + W / 2, c + 1 - W / 2
            xm = (x0 + x1) / 2
            path = (f"M {x0},{a0} C {xm},{a0} {xm},{b0} {x1},{b0} L {x1},{b1} "
                    f"C {xm},{b1} {xm},{a1} {x0},{a1} Z")
            shapes.append(dict(type="path", path=path, fillcolor=_rgba(scope_color(ref_mod[s]), 0.45),
                               line=dict(width=0), layer="below"))
    for c, blk in enumerate(blocks):
        for k, (y0, y1, g) in enumerate(blk, 1):
            shapes.append(dict(type="rect", x0=c - W / 2, x1=c + W / 2, y0=y0, y1=y1, fillcolor="#333333",
                               line=dict(color="white", width=1)))
            hx.append(c); hy.append((y0 + y1) / 2)
            ev = sum(L.n_events[s] for s in g)
            ht.append(f"<b>Markov time {mts[c]:g}, module {k}</b><br>{ev:,} events<br>" + "<br>".join(sorted(g)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=18, color="rgba(0,0,0,0)"),
                             hovertext=ht, hoverinfo="text", showlegend=False))
    for s in sigs:
        y0, y1 = seg[0][s]
        fig.add_annotation(x=-W / 2 - 0.04, y=(y0 + y1) / 2, text=f"{s.strip('{}')} ({L.n_events[s]:,})", showarrow=False,
                           xanchor="right", font=dict(size=10, color=TEXT))
    ymax = max(max(p[1] for p in sg.values()) for sg in seg) + gap
    fig.update_layout(shapes=shapes, height=height, margin=dict(l=10, r=10, t=10, b=50),
                      xaxis=dict(range=[-2.4, len(mts) - 1 + W / 2 + 0.1], tickvals=list(range(len(mts))),
                                 ticktext=[f"<b>{mt:g}</b>" + (" ◀" if mt == chosen else "") + f"<br>{len(results[mt])} mod."
                                           for mt in mts], showgrid=False, zeroline=False, title="Markov time"),
                      yaxis=dict(range=[ymax, -gap], visible=False), plot_bgcolor="white", paper_bgcolor="white",
                      hoverlabel=dict(bgcolor="white"))
    return fig


def _rgba(hex_, a):
    r, g, b = (int(hex_[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{a})"


# ---------------------------------------------------------------- signal bars
def signal_bars(raw, weights=None, b=None):
    keys = ["participants", "multiplicity", "wait", "pace_change"]
    vals = [raw[k] for k in keys]
    fig = go.Figure(go.Bar(y=[SIGNAL_LABEL[k] + (f"  (w {weights[k]:g})" if weights else "") for k in keys], x=vals,
                           orientation="h", marker=dict(color=["#257de0" if not weights or weights[k] > 0 else "#cfcfcf" for k in keys]),
                           text=[f"{v:.2f}" for v in vals], textposition="outside", hoverinfo="skip"))
    if b is not None:
        fig.add_vline(x=b, line=dict(color="#e2621f", width=2, dash="dash"))
        fig.add_annotation(x=b, y=1.08, yref="paper", text=f"b = {b:.3f}", showarrow=False, font=dict(color="#a8460f", size=11))
    fig.update_layout(height=220, margin=dict(l=10, r=30, t=26, b=20), xaxis=dict(range=[0, 1.12], gridcolor=GRID),
                      yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white", bargap=0.45)
    return fig


def f1_curve(rows, gt_k=None, height=300):
    ks = [r["k"] for r in rows]; fs = [r["macro F1"] for r in rows]
    fig = go.Figure(go.Scatter(x=ks, y=fs, mode="lines+markers", line=dict(color="#257de0", width=2),
                               marker=dict(size=8), hovertemplate="k = %{x}<br>macro F1 %{y:.3f}<extra></extra>", name="merge"))
    if gt_k:
        fig.add_vline(x=gt_k, line=dict(color=TEXT2, dash="dash", width=1))
        fig.add_annotation(x=gt_k, y=0.05, text=f"GT k = {gt_k}", showarrow=False, xanchor="left", font=dict(color=TEXT2, size=10))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=40), xaxis=dict(title="number of scopes k", dtick=1, gridcolor=GRID),
                      yaxis=dict(title="macro F1", range=[0, 1.02], gridcolor=GRID), plot_bgcolor="white", paper_bgcolor="white")
    return fig
