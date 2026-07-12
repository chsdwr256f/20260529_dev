import streamlit as st

st.set_page_config(
    page_title="The University of Edinburgh Information Navigator",
    layout="wide"
)

st.title("The University of Edinburgh Information Navigator")

st.markdown("""
This tool uses a knowledge graph to support information navigation across
University of Edinburgh webpages.

Please use the pages in the sidebar:

- **Ask the graph**: ask information-seeking questions.
- **Browse entities**: inspect entities and their relationships.
- **Evaluation**: Record task outcomes and provide feedback.
- **Feedback**: Provide general feedback.
""")