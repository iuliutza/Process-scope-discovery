"""Checks the engine against the notebook outputs on the official Logistics log (run from the tool folder:
python tests/check_logistics.py)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import engine as E

L = E.load_log("data/logistics.json")
S = E.signals(L, E.build_graph(L), 0.0)
B = E.scores(S, E.DEFAULTS["weights"], 0.0)
M = E.merge(L, S, B)
gts, sev = E.gt_sets(L, json.load(open("data/gt_logistics.json"))), E.sig_events(L)
expected = {12: 0.7705, 11: 0.6307, 10: 0.6307, 9: 0.7053, 8: 0.6739, 7: 0.5564, 6: 0.5442, 5: 0.5897, 4: 0.5203,
            3: 0.4867, 2: 0.4774, 1: 0.3584}                     # notebook f1_curve_cell output
for k, f in expected.items():
    got = E.score(M.cut(M.n_merges_for(k=k)), gts, sev)[1]
    assert abs(got - f) < 1e-3, (k, got, f)
assert abs(M.merges[0][0] - 0.111) < 1e-3
parts, _ = E.run_infomap(L, S, B, 0.75)
assert len(parts) == 4
procs, bad = E.make_rules(L, E.scope_map(E.by_size(L, M.cut(M.n_merges_for(k=4)))))
assert not bad
print("all checks passed")
