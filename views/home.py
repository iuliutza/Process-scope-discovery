import streamlit as st

import common as C

st.markdown("""
<style>
.card {border:1px solid #e3e5e8; border-radius:10px; padding:22px 24px 10px 24px; height:100%; background:#fbfbfc}
.card h3 {margin-top:0}
.muted {color:#666666; font-size:0.95rem}
</style>""", unsafe_allow_html=True)

st.title("Where does one process end and the next begin?")
st.markdown(
    "<p class='muted'>A tool for finding <b>process scopes</b> in object-centric event logs (OCEL 2.0). "
    "It proposes borders between detected scopes with two algorithms, <b>hierarchical merging</b> and <b>Infomap</b>, "
    "and writes them as Procellar rules. The tool suggests scopes rules, naming and defining the processes stays with you.</p>",
    unsafe_allow_html=True)
st.write("")

with st.expander("How the two algorithms differ, in one paragraph"):
    st.markdown(
        "Both start from the same graph: **signatures** (the set of object types an event touches) as nodes and "
        "**hand-overs** (an object moving from one event to its next) as edges. Each edge gets a **boundary score b** "
        "from four signals: does the set of object types change, does batching change, does the object wait, does "
        "the pace change. **Hierarchical merging** joins the two groups with the lowest boundary score, again and again, "
        "and you cut the resulting tree at the number of scopes you want. **Infomap** follows the flow of objects instead: "
        "it looks for groups a random walker rarely leaves, using hand-overs weighted by how much they stay together, and "
        "you choose the zoom level.")

c1, c2 = st.columns(2, gap="large")
with c1:
    st.markdown("""<div class='card'><h3>📘 Learn how it works</h3>
<p>A guided walk through both algorithms on the <a href='""" + C.LOGISTICS_URL + """' target='_blank'>Logistics</a> log (35,372 events, 7 object types).</p>
<ul><li>what a signature and a hand-over are</li>
<li>the four signals that point at a border, and the boundary score</li>
<li>the merge step by step, and Infomap's zoom levels</li>
<li>both methods next to the documented phases of the log</li></ul>
<p class='muted'>Switch on <i>free explore</i> to change weights and levels yourself.</p></div>""",
                unsafe_allow_html=True)
    st.write("")
    st.page_link("views/explainer.py", label="Start the walkthrough", icon=":material/arrow_forward:")
with c2:
    st.markdown("""<div class='card'><h3>🛠️ Run on my log</h3>
<p>Load your own OCEL 2.0 JSON file and get scopes rules.</p>
<ul><li>overview of the signals on your log, before you choose weights</li>
<li>hierarchical merge (pick k or a height) and Infomap (pick a Markov time)</li>
<li>compare both methods, optionally against your own ground-truth rules</li>
<li>export Procellar rules and a short report</li></ul>
<p class='muted'>From raw log to Procellar rules in a few clicks.</p></div>""", unsafe_allow_html=True)
    st.write("")
    st.page_link("views/workbench.py", label="Open the workbench", icon=":material/arrow_forward:")



st.write("")
st.divider()
p1, p2 = st.columns([0.6, 0.4], gap="large")
with p1:
    st.markdown("#### Next step: Procellar")
    st.markdown("<p class='muted'>The rules this tool exports are Procellar definition files. Import them in "
                "Procellar to check, edit and name the scopes, and to write them into your OCEL as process objects. "
                "Procellar's repository explains how to install and start it.</p>",
                unsafe_allow_html=True)
with p2:
    st.write("")
    st.link_button("Procellar on GitHub", C.PROCELLAR_REPO, icon=":material/open_in_new:", type="primary", width="stretch")
st.write("")

