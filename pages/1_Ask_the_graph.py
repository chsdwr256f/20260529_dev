import pandas as pd
import streamlit as st

from utils import (
    get_graph_and_entities,
    get_openai_client,
    classify_question_topic,
    retrieve_relevant_entities,
    retrieve_mentioned_entities,
    retrieve_relevant_triples,
    triples_to_text,
    matched_entities_to_text,
    ask_llm,
    build_evidence_graph,
    draw_interactive_pyvis,
    shorten_uri,
)

st.set_page_config(
    page_title="Ask the graph",
    layout="wide"
)

st.title("Ask the graph")

graph, entities_df = get_graph_and_entities()
client, openai_available, openai_error = get_openai_client()

with st.sidebar:
    st.header("System status")
    st.write(f"LLM available: {openai_available}")
    st.success(f"Graph loaded. Total triples: {len(graph):,}")

if "user_question" not in st.session_state:
    st.session_state["user_question"] = ""


def clear_question():
    st.session_state["user_question"] = ""

    for key in ["evidence_rows", "matched_entities", "answer"]:
        st.session_state.pop(key, None)


col1, col2 = st.columns([5, 1])

with col1:
    user_question = st.text_input(
        "Ask a question about the university",
        key="user_question"
    )

with col2:
    st.write("")
    st.write("")
    st.button("Clear", on_click=clear_question)

if st.button("Answer question"):
    if not user_question.strip():
        st.warning("Please enter a question.")

    else:
        with st.spinner("Retrieving evidence from the knowledge graph..."):

            topic_result = classify_question_topic(user_question)

            answer_types = topic_result["answer_types"]
            mentioned_entities = topic_result["mentioned_entities"]
            retrieval_concepts = topic_result["retrieval_concepts"]

            matched_entities = retrieve_relevant_entities(
                entities_df,
                user_question,
                answer_types=answer_types,
                retrieval_concepts=retrieval_concepts,
                top_k=10
            )

            supporting_entities = retrieve_mentioned_entities(
                entities_df,
                mentioned_entities,
                top_k=10
            )

            evidence_entities = pd.concat(
                [supporting_entities, matched_entities],
                ignore_index=True
            ).drop_duplicates(subset=["uri"])

            evidence_rows = retrieve_relevant_triples(
                graph,
                evidence_entities
            )

            context_text = triples_to_text(evidence_rows)
            matched_entities_text = matched_entities_to_text(matched_entities)

            answer, error = ask_llm(
                user_question,
                matched_entities_text,
                context_text
            )

            st.session_state["evidence_rows"] = evidence_rows
            st.session_state["matched_entities"] = matched_entities
            st.session_state["answer"] = answer
            st.session_state["answer_error"] = error
            st.session_state["topic_result"] = topic_result

if "answer" in st.session_state:
    st.markdown("### Answer")

    if st.session_state.get("answer_error"):
        st.error(st.session_state["answer_error"])

    elif st.session_state["answer"]:
        st.write(st.session_state["answer"])

    else:
        st.info("No answer was returned.")

    topic_result = st.session_state.get("topic_result", {})

    with st.expander("Retrieval classification", expanded=False):
        st.write("Answer types:", topic_result.get("answer_types", []))
        st.write("Mentioned entities:", topic_result.get("mentioned_entities", []))
        st.write("Retrieval concepts:", topic_result.get("retrieval_concepts", []))

    with st.expander("Entities retrieval", expanded=False):
        matched_entities = st.session_state.get("matched_entities")

        if matched_entities is None or matched_entities.empty:
            st.info("No relevant entities were retrieved from the graph.")
        else:
            display_df = matched_entities[
                ["label", "type", "source", "match_score"]
            ].copy()

            st.dataframe(
                display_df,
                width="stretch",
                height=220
            )

    evidence_rows = st.session_state.get("evidence_rows", [])
    evidence_df = pd.DataFrame(evidence_rows)

    with st.expander(f"Evidence triples ({len(evidence_df)})", expanded=False):
        if evidence_df.empty:
            st.info("No evidence triples available.")
        else:
            triples_display = evidence_df.copy()

            for col in ["subject", "predicate", "object"]:
                triples_display[col] = triples_display[col].apply(shorten_uri)

            st.dataframe(
                triples_display,
                width="stretch",
                height=250
            )

    st.markdown("### Evidence knowledge graph")

    if evidence_rows:
        if st.button("Show evidence graph"):
            evidence_graph = build_evidence_graph(evidence_rows)

            if evidence_graph.number_of_edges() == 0:
                st.info("No graph structure available.")
            else:
                draw_interactive_pyvis(evidence_graph)
    else:
        st.info("No evidence graph available.")