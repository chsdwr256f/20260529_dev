import streamlit as st

from rdflib.namespace import DCTERMS

from utils import (
    get_graph_and_entities,
    search_entities,
    get_node_from_selection,
    get_label,
    classify_node,
    get_comment,
    describe_entity,
    shorten_uri,
    build_ego_network,
    draw_interactive_pyvis,
)

st.set_page_config(
    page_title="Browse entities",
    layout="wide"
)

st.title("Browse entities")

graph, entities_df = get_graph_and_entities()

with st.sidebar:
    st.header("System status")
    st.success(f"Graph loaded. Total triples: {len(graph):,}")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Entity search")

    query = st.text_input(
        "Search by label, URI, or type",
        key="browse_query"
    )

    results_df = search_entities(entities_df, query)

    if results_df.empty:
        st.warning("No matching entities found.")
        st.stop()

    results_df = results_df.copy()
    results_df["label"] = results_df["label"].fillna("").astype(str)
    results_df["type"] = results_df["type"].fillna("Unclassified").astype(str)
    results_df["uri"] = results_df["uri"].fillna("").astype(str)

    results_df["display"] = (
        results_df["label"]
        + "  |  "
        + results_df["type"]
        + "  |  "
        + results_df["uri"].map(shorten_uri)
    )

    selection = st.selectbox(
        "Select an entity",
        results_df["display"].tolist(),
        key="browse_selection"
    )

    selected_node = get_node_from_selection(results_df, selection)

    if selected_node is None:
        st.warning("Could not resolve the selected entity.")
        st.stop()

    # Removed the Matching entities dataframe here.
    # This makes the page cleaner and avoids rendering a large table.

with col2:
    st.subheader("Entity details")

    entity_label = get_label(graph, selected_node)
    entity_type = classify_node(graph, selected_node)
    entity_comment = get_comment(graph, selected_node)

    st.markdown(f"**Label:** {entity_label}")
    st.markdown(f"**URI:** `{selected_node}`")
    st.markdown(f"**Type:** {entity_type}")

    sources = sorted(set(
        str(s)
        for s in graph.objects(selected_node, DCTERMS.source)
        if str(s).startswith("http")
    ))

    st.markdown("**Original source links:**")

    if sources:
        for url in sources:
            st.markdown(f"- [{url}]({url})")
    else:
        st.info("No source links found.")

    if entity_comment:
        st.markdown(f"**Description:** {entity_comment}")

    outgoing_df, incoming_df = describe_entity(graph, selected_node)

    tab1, tab2, tab3 = st.tabs(
        ["Outgoing relations", "Incoming relations", "Neighbourhood graph"]
    )

    with tab1:
        if outgoing_df.empty:
            st.info("No outgoing relations available.")
        else:
            st.dataframe(
                outgoing_df,
                width="stretch",
                height=350
            )

    with tab2:
        if incoming_df.empty:
            st.info("No incoming relations available.")
        else:
            st.dataframe(
                incoming_df,
                width="stretch",
                height=350
            )

    with tab3:
        if st.button("Show neighbourhood graph"):
            ego = build_ego_network(graph, selected_node)

            if ego.number_of_edges() == 0:
                st.info("No graph neighbourhood available.")
            else:
                draw_interactive_pyvis(ego)