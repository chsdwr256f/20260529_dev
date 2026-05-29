import re
import sys
import json
import time
import uuid
import tempfile
from pathlib import Path
from datetime import datetime

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS, DCTERMS, SKOS

from openai import OpenAI

try:
    from pyvis.network import Network
    PYVIS_AVAILABLE = True
except Exception:
    PYVIS_AVAILABLE = False


# -----------------------------
# Paths and loading
# -----------------------------

def get_base_dir():
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


BASE_DIR = get_base_dir()
TTL_FILE_PATH = BASE_DIR / "kg_with_instances.ttl"


@st.cache_resource
def get_openai_client():
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        return client, True, None
    except Exception as e:
        return None, False, str(e)


@st.cache_resource(show_spinner="Loading knowledge graph...")
def load_graph_from_file(local_path=TTL_FILE_PATH):
    g = Graph()
    g.parse(local_path, format="turtle")
    return g


@st.cache_resource(show_spinner="Building entity index...")
def load_entities_from_graph():
    graph = load_graph_from_file()
    entities_df = list_entity_candidates(graph)
    return entities_df


def get_graph_and_entities():
    graph = load_graph_from_file()
    entities_df = load_entities_from_graph()
    return graph, entities_df


# -----------------------------
# RDF helpers
# -----------------------------

COMMON_LABEL_PROPS = [
    RDFS.label,
    SKOS.prefLabel,
    DCTERMS.title
]

COMMON_DESC_PROPS = [
    RDFS.comment,
    DCTERMS.description,
    SKOS.definition,
    SKOS.scopeNote
]


def shorten_uri(uri):
    text = str(uri)
    if "#" in text:
        return text.split("#")[-1]
    if "/" in text:
        return text.rstrip("/").split("/")[-1]
    return text


def get_label(g, node):
    for prop in COMMON_LABEL_PROPS:
        for obj in g.objects(node, prop):
            if isinstance(obj, Literal):
                return str(obj)
    if isinstance(node, URIRef):
        return shorten_uri(node)
    return str(node)


def get_comment(g, node):
    for prop in COMMON_DESC_PROPS:
        for obj in g.objects(node, prop):
            if isinstance(obj, Literal):
                return str(obj)
    return ""


def classify_node(g, node):
    types = [
        get_label(g, t)
        for t in g.objects(node, RDF.type)
        if t != OWL.NamedIndividual
    ]
    return ", ".join(types) if types else "Unclassified"


def get_source_urls(g, node):
    sources = []
    for src in g.objects(node, DCTERMS.source):
        src = str(src)
        if src.startswith("http"):
            sources.append(src)
    return sorted(set(sources))


def list_entity_candidates(g):
    candidates = []

    for s in set(g.subjects()):
        if isinstance(s, Literal):
            continue

        source_urls = get_source_urls(g, s)

        candidates.append({
            "label": get_label(g, s),
            "uri": str(s),
            "source": source_urls[0] if source_urls else "",
            "sources": source_urls,
            "type": classify_node(g, s),
        })

    if not candidates:
        return pd.DataFrame(columns=["label", "uri", "source", "sources", "type"])

    df = pd.DataFrame(candidates)

    for col in ["label", "uri", "source", "sources", "type"]:
        if col not in df.columns:
            df[col] = "" if col != "sources" else [[]]

    return df.sort_values(["label", "type"]).reset_index(drop=True)


# -----------------------------
# Entity search
# -----------------------------

def search_entities(df, query):
    if df.empty:
        return df

    if not query:
        return df.head(100)

    mask = (
        df["label"].str.contains(query, case=False, na=False)
        | df["uri"].str.contains(query, case=False, na=False)
        | df["type"].str.contains(query, case=False, na=False)
    )

    return df.loc[mask].reset_index(drop=True)


def get_node_from_selection(df, selection_label):
    if df.empty or "display" not in df.columns or "uri" not in df.columns:
        return None

    row = df.loc[df["display"] == selection_label]

    if row.empty:
        return None

    uri = str(row.iloc[0]["uri"]).strip()

    if not uri:
        return None

    return URIRef(uri)


def describe_entity(g, node):
    outgoing = []
    incoming = []

    for p, o in g.predicate_objects(node):
        outgoing.append({
            "predicate": get_label(g, p),
            "object": get_label(g, o),
            "object_uri": str(o) if isinstance(o, URIRef) else "",
        })

    for s, p in g.subject_predicates(node):
        incoming.append({
            "subject": get_label(g, s),
            "subject_uri": str(s) if isinstance(s, URIRef) else "",
            "predicate": get_label(g, p),
        })

    return pd.DataFrame(outgoing), pd.DataFrame(incoming)


# -----------------------------
# Retrieval
# -----------------------------

def normalise_query_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def score_entity_match(query, row):
    query_norm = normalise_query_text(query)
    label_norm = normalise_query_text(row.get("label", ""))
    type_norm = normalise_query_text(row.get("type", ""))
    uri_norm = normalise_query_text(row.get("uri", ""))

    if not query_norm:
        return 0

    score = 0
    q_tokens = set(query_norm.split())
    label_tokens = set(label_norm.split())
    type_tokens = set(type_norm.split())
    uri_tokens = set(uri_norm.split())

    score += len(q_tokens & label_tokens) * 4
    score += len(q_tokens & type_tokens) * 2
    score += len(q_tokens & uri_tokens)

    if query_norm in label_norm:
        score += 10

    if query_norm in uri_norm:
        score += 4

    return float(score)


def retrieve_relevant_entities(
    entities_df,
    question,
    answer_types=None,
    retrieval_concepts=None,
    top_k=10
):
    empty_result = pd.DataFrame(
        columns=["label", "uri", "type", "match_score"]
    )

    if entities_df.empty:
        return empty_result

    working = entities_df.copy()

    if answer_types:
        answer_types_lower = [
            str(t).strip().lower()
            for t in answer_types
        ]

        working = working.loc[
            working["type"].astype(str).str.lower().isin(answer_types_lower)
        ].copy()

    if working.empty:
        return empty_result

    working["match_score"] = working.apply(
        lambda r: score_entity_match(question, r),
        axis=1
    ).astype(float)

    question_terms = [
        w.lower()
        for w in re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", question)
        if w.lower() not in {
            "which", "what", "where", "when", "who", "how",
            "the", "and", "for", "with", "within", "about",
            "student", "students", "take", "list", "show",
            "find", "give", "does", "are", "is"
        }
    ]

    for term in question_terms:
        working.loc[
            working["label"].astype(str).str.lower().str.contains(
                re.escape(term),
                na=False
            ),
            "match_score"
        ] += 30

    if retrieval_concepts:
        search_text = working["label"].astype(str)

        if "comment" in working.columns:
            search_text = search_text + " " + working["comment"].astype(str)

        if "uri" in working.columns:
            search_text = search_text + " " + working["uri"].astype(str)

        search_text = search_text.str.lower()

        for concept in retrieval_concepts:
            concept = str(concept).strip().lower()

            if concept:
                working.loc[
                    search_text.str.contains(re.escape(concept), na=False),
                    "match_score"
                ] += 100

    working = (
        working.loc[working["match_score"] > 0]
        .sort_values("match_score", ascending=False)
    )

    if working.empty:
        return empty_result

    return working.head(top_k).reset_index(drop=True)


def retrieve_mentioned_entities(entities_df, mentioned_entities, top_k=10):
    if not mentioned_entities:
        return pd.DataFrame(columns=entities_df.columns)

    results = []

    for ent in mentioned_entities:
        ent_lower = str(ent).strip().lower()

        matches = entities_df[
            entities_df["label"].astype(str).str.lower().eq(ent_lower)
        ].copy()

        if matches.empty:
            matches = entities_df[
                entities_df["label"].astype(str).str.lower().str.contains(
                    re.escape(ent_lower),
                    na=False
                )
            ].copy()

        if not matches.empty:
            results.append(matches.head(top_k))

    if not results:
        return pd.DataFrame(columns=entities_df.columns)

    return (
        pd.concat(results, ignore_index=True)
        .drop_duplicates(subset=["uri"])
        .head(top_k)
    )


def retrieve_relevant_triples(
    graph,
    matched_entities,
    max_outgoing=8,
    max_incoming=8
):
    if matched_entities is None or matched_entities.empty:
        return []

    metadata_predicates = {
        str(RDFS.label),
        str(RDFS.comment),
        str(RDF.type),
        str(SKOS.prefLabel),
        str(DCTERMS.title),
        str(DCTERMS.description),
    }

    evidence_rows = []

    for _, row in matched_entities.iterrows():
        uri = row.get("uri")

        if pd.isna(uri):
            continue

        node = URIRef(str(uri))

        outgoing_count = 0

        for s, p, o in graph.triples((node, None, None)):
            if str(p) in metadata_predicates:
                continue

            if outgoing_count >= max_outgoing:
                break

            evidence_rows.append({
                "subject": get_label(graph, s),
                "predicate": get_label(graph, p),
                "object": get_label(graph, o),
                "object_uri": str(o) if isinstance(o, URIRef) else ""
            })

            outgoing_count += 1

        incoming_count = 0

        for s, p, o in graph.triples((None, None, node)):
            if str(p) in metadata_predicates:
                continue

            if incoming_count >= max_incoming:
                break

            evidence_rows.append({
                "subject": get_label(graph, s),
                "predicate": get_label(graph, p),
                "object": get_label(graph, o),
                "object_uri": str(o) if isinstance(o, URIRef) else ""
            })

            incoming_count += 1

    return evidence_rows


def triples_to_text(evidence_rows, max_rows=80):
    if not evidence_rows:
        return ""

    lines = []

    for row in evidence_rows[:max_rows]:
        lines.append(
            f"{row.get('subject', '')} | "
            f"{row.get('predicate', '')} | "
            f"{row.get('object', '')}"
        )

    return "\n".join(lines)


def matched_entities_to_text(matched_entities, max_rows=10):
    if matched_entities is None or matched_entities.empty:
        return ""

    lines = []

    for _, row in matched_entities.head(max_rows).iterrows():
        lines.append(
            f"Label: {row.get('label', '')} | "
            f"Type: {row.get('type', '')} | "
            f"URI: {row.get('uri', '')} | "
            f"Match score: {row.get('match_score', '')}"
        )

    return "\n".join(lines)

def is_safe_select_sparql(query):
    q = query.strip().lower()

    blocked = [
        "insert", "delete", "drop", "clear", "create",
        "load", "copy", "move", "add", "service"
    ]

    if not q.startswith("prefix") and not q.startswith("select"):
        return False

    if "select" not in q:
        return False

    return not any(word in q for word in blocked)

def generate_sparql_from_question(question):
    ontology_summary = """
Classes:
Programme, Course, School, Department, Staff, Person, Student,
Research, ResearchProject, ResearchCentre, Policy, Document,
Event, Scholarship, ContactPoint, OrganisationUnit, Topic, WebPage

Object properties:
offersProgramme, offersCourse, teachesCourse, hasPrerequisite,
hasContactPoint, hasScholarship, memberOf, relatedToTopic,
containedInDocument, governedByPolicy, eligibleForCourse,
availableInYear

Common predicates:
rdf:type, rdfs:label, rdfs:comment, dcterms:source

Namespaces:
ont: https://www.ed.ac.uk/ontology/organisation#
kg: https://www.ed.ac.uk/kg/
"""

    prompt = f"""
You are generating SPARQL for a University of Edinburgh RDF knowledge graph.

Use only the ontology below:
{ontology_summary}

Task:
Convert the user question into one safe SPARQL SELECT query.

Strict rules:
- Return SPARQL only.
- Use SELECT only.
- Do not use INSERT, DELETE, DROP, LOAD, SERVICE, UPDATE, or CONSTRUCT.
- Prefer rdfs:label and rdfs:comment keyword matching if exact relationships may not exist.
- Use FILTER(CONTAINS(LCASE(STR(?label)), "...")) for keyword matching.
- Include dcterms:source if useful.
- Use LIMIT 20.
- Do not invent new classes or properties.
- Only restrict rdf:type when highly certain.
- If uncertain, use variable type matching:
    ?entity rdf:type ?type
- Avoid overly restrictive queries that may miss valid entities.

User question:
{question}
"""

    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=prompt,
            temperature=0
        )

        sparql = response.output_text.strip()

        sparql = (
            sparql
            .replace("```sparql", "")
            .replace("```", "")
            .strip()
        )

        if not is_safe_select_sparql(sparql):
            return None, "Generated query was blocked because it was not a safe SELECT query."

        return sparql, None

    except Exception as e:
        return None, str(e)


def run_generated_sparql(graph, question):
    sparql_query, error = generate_sparql_from_question(question)

    if error:
        return None, None, error

    try:
        results = graph.query(sparql_query)

        rows = []
        for row in results:
            rows.append({
                str(var): str(value)
                for var, value in zip(results.vars, row)
            })

        return pd.DataFrame(rows), sparql_query, None

    except Exception as e:
        return None, sparql_query, str(e)

def sparql_results_to_entities_df(sparql_df):
    if sparql_df is None or sparql_df.empty:
        return pd.DataFrame(columns=["label", "type", "uri", "match_score"])

    uri_cols = ["uri", "entity", "project", "course", "programme", "person", "subject"]
    label_cols = ["label", "entityLabel", "projectLabel", "courseLabel", "programmeLabel"]

    rows = []

    for _, row in sparql_df.iterrows():
        uri = ""

        for col in uri_cols:
            if col in sparql_df.columns and str(row.get(col, "")).startswith("http"):
                uri = row.get(col, "")
                break

        if not uri:
            continue

        label = ""
        for col in label_cols:
            if col in sparql_df.columns and row.get(col, ""):
                label = row.get(col, "")
                break

        rows.append({
            "label": label if label else uri.split("/")[-1],
            "type": "SPARQL result",
            "uri": uri,
            "match_score": None
        })

    return pd.DataFrame(rows)


# -----------------------------
# LLM
# -----------------------------

TOPICS = [
    "course_eligibility",
    "course",
    "programme",
    "requirement",
    "staff",
    "research",
    "policy",
    "scholarship",
    "contact",
    "event",
    "organisation_unit",
    "document",
    "general"
]


def classify_question_topic(question):
    client, available, error = get_openai_client()

    if not available:
        return {
            "answer_types": [],
            "mentioned_entities": [],
            "retrieval_concepts": []
        }

    prompt = f"""
You are classifying questions for a university knowledge graph.

Identify:

1. answer_types: The main type of answer the user wants (must select from topic list). 
For example:
- "which courses" -> "course"
- "who teaches" -> "staff"
- "what requirements" -> "requirement"

2. mentioned_entities: Specific named entities explicitly mentioned or domain topics when they are the object of the search. These are important retrieval anchors.
Examples: Chemistry 1A, School of Informatics

3. retrieval_concepts: Up to 3 important semantic concepts useful for ranking and graph traversal.
- prefer single words, maximum two words
- do not include entity types
- Examples: teaching, semester, eligibility, contact

Select topics from this list only:
{TOPICS}

Important rules:
- A question may involve multiple ontology classes.
- Mentioned entities are usually more important than broad topics.

Return JSON only:
{{
  "answer_types": "...",
  "mentioned_entities": ["...", "..."],
  "concepts": ["...", "..."]
}}

Example:

Question:
Which staff members are teaching Chemistry 1A this semester?

Output:
{{
  "answer_types": ["Staff"],
  "mentioned_entities": ["Chemistry 1A"],
  "retrieval_concepts": ["teaching", "semester"]
}}

Question:
{question}

"""

    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=prompt,
            temperature=0
        )

        result = json.loads(response.output_text)

        return {
            "answer_types": result.get("answer_types", []),
            "mentioned_entities": result.get("mentioned_entities", []),
            "retrieval_concepts": result.get("retrieval_concepts", [])
        }

    except Exception as e:
        st.warning(f"Topic classification failed: {e}")

        return {
            "answer_types": [],
            "mentioned_entities": [],
            "retrieval_concepts": []
        }


def ask_llm(question, matched_entities_text, triples_context):
    client, available, error = get_openai_client()

    if not available:
        return None, f"OpenAI unavailable: {error}"

    prompt = f"""
You are assisting users with a university domain question.
Answer the user's question ONLY using the provided knowledge graph context.

Rules:
- Do not invent information.
- If the KG contains direct evidence, answer clearly.
- If the KG does NOT contain enough information for a definitive answer:
    - say that the information is not fully available,
    - BUT provide the most relevant related entities, webpages, or source links.
- If eligibility, requirements, or regulations cannot be confirmed, direct the user to the relevant programme or course pages.
- Prefer helping the user navigate to relevant information rather than refusing the question.
- Mention related courses, programmes, schools, or policies if useful.
- Course pages would contain information including school, college, level, credits, availability for who, description, Pre-requisites, assessment, exam, contact information.
- Keep answers concise and factual.

Evidence triples:
{triples_context}

Matched entities:
{matched_entities_text}

Question:
{question}

"""

    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=prompt,
            temperature=0
        )

        return response.output_text, None

    except Exception as e:
        return None, str(e)


# -----------------------------
# Graph visualisation
# -----------------------------

def is_metadata_predicate(predicate):
    metadata_predicates = {
        str(RDFS.label),
        str(RDFS.comment),
        str(RDF.type),
        str(DCTERMS.source),
    }
    return str(predicate) in metadata_predicates


def build_ego_network(g, center_node, max_edges=60):
    G = nx.DiGraph()

    center_label = get_label(g, center_node)
    G.add_node(str(center_node), label=center_label, kind="center")

    edge_count = 0

    for p, o in g.predicate_objects(center_node):
        if edge_count >= max_edges:
            break

        if is_metadata_predicate(p):
            continue

        obj_key = str(o)
        G.add_node(obj_key, label=get_label(g, o), kind="out")
        G.add_edge(str(center_node), obj_key, label=get_label(g, p))
        edge_count += 1

    for s, p in g.subject_predicates(center_node):
        if edge_count >= max_edges:
            break

        if is_metadata_predicate(p):
            continue

        sub_key = str(s)
        G.add_node(sub_key, label=get_label(g, s), kind="in")
        G.add_edge(sub_key, str(center_node), label=get_label(g, p))
        edge_count += 1

    return G


def build_evidence_graph(evidence_rows):
    G = nx.DiGraph()

    excluded_predicates = {
        "label",
        "comment",
        "type",
        "source",
        "prefLabel",
        "title",
        "description"
    }

    for row in evidence_rows:
        s = str(row["subject"])
        p = str(row["predicate"])
        o = str(row["object"])

        if p in excluded_predicates:
            continue

        if s == o:
            continue

        G.add_node(s, label=s)
        G.add_node(o, label=o)
        G.add_edge(s, o, label=p)

    return G


def clean_graph_label(x):
    x = str(x)

    if "#" in x:
        x = x.split("#")[-1]
    else:
        x = x.rstrip("/").split("/")[-1]

    x = x.replace("_", " ").replace("-", " ")

    return x[:80]


def draw_interactive_pyvis(graph_nx, height="700px", width="100%"):
    if not PYVIS_AVAILABLE:
        st.info("Pyvis is not installed. Run: pip install pyvis")
        return

    with st.expander("Graph layout settings", expanded=False):
        physics_enabled = st.toggle("Enabled", value=True)
        solver = st.selectbox(
            "Layout algorithm",
            ["barnesHut", "forceAtlas2Based", "repulsion", "hierarchicalRepulsion"],
            index=0
        )

    net = Network(height=height, width=width, directed=True)

    for node, data in graph_nx.nodes(data=True):
        label = clean_graph_label(data.get("label", str(node)))
        net.add_node(str(node), label=label, title=label)

    for source, target, data in graph_nx.edges(data=True):
        label = clean_graph_label(data.get("label", ""))
        net.add_edge(str(source), str(target), label=label)

    if solver == "barnesHut":
        net.barnes_hut()
    elif solver == "forceAtlas2Based":
        net.force_atlas_2based()
    elif solver == "repulsion":
        net.repulsion()
    elif solver == "hierarchicalRepulsion":
        net.hrepulsion()

    net.toggle_physics(physics_enabled)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        net.save_graph(tmp_file.name)
        html = Path(tmp_file.name).read_text(encoding="utf-8")

    components.html(html, height=750, scrolling=True)