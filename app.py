"""Scope tool: run with  streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="ScopeFinder", page_icon="🧭", layout="wide")
st.logo("assets/logo.png", size="large")

home = st.Page("views/home.py", title="Home", icon=":material/home:", default=True)
explainer = st.Page("views/explainer.py", title="Learn how it works", icon=":material/school:")
workbench = st.Page("views/workbench.py", title="Run on my log", icon=":material/build:")
nav = st.navigation([home, explainer, workbench])
nav.run()
