import uuid
from datetime import datetime

import pandas as pd
import streamlit as st
import gspread

from google.oauth2.service_account import Credentials


st.set_page_config(
    page_title="Evaluation",
    layout="wide"
)

st.title("Evaluation")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]


# -------------------------
# Consent Gate
# -------------------------

if "evaluation_consent_given" not in st.session_state:
    st.session_state["evaluation_consent_given"] = False

if not st.session_state["evaluation_consent_given"]:
    show_consent_dialog()
    st.stop()

@st.cache_resource
def connect_to_gsheet():
    credentials = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=SCOPES
    )

    client = gspread.authorize(credentials)

    sheet = client.open(
        st.secrets["google_sheet"]["sheet_name"]
    )

    worksheet = sheet.worksheet(
        st.secrets["google_sheet"]["worksheet_name"]
    )

    return worksheet

# -----------------------------
# Participant and task details
# -----------------------------

st.header("User Information")

user_name = st.text_input(
    "Participant Name",
    placeholder="Dummy",
    key="eval_user_id"
)

user_type = st.radio(
    "Please select your user type:",
    [
        "Prospective student",
        "Current student",
        "Staff",
        "External stakeholder"
    ],
    key="eval_user_type"
)

task_id = st.text_input(
    key="eval_task_id"
)

# -----------------------------
# Tool feedback
# -----------------------------

st.markdown("---")
st.header("Tool Feedback")

found_info = st.radio(
    "Were you able to find the information?",
    ["Yes", "No"],
    index=None,
    key="eval_found_info"
)

steps_taken = st.number_input(
    "Number of steps taken",
    min_value=0,
    step=1,
    key="eval_steps_taken"
)

st.markdown("#### Time Taken")

col1, col2 = st.columns(2)

with col1:
    tool_minutes = st.number_input(
        "Minutes",
        min_value=0,
        step=1,
        key="tool_minutes"
    )

with col2:
    tool_seconds = st.number_input(
        "Seconds",
        min_value=0,
        max_value=59,
        step=1,
        key="tool_seconds"
    )

tool_total_seconds = (
    tool_minutes * 60
    + tool_seconds
)

feedback_comment = st.text_area(
    "Comments",
    placeholder="Comments (optional)",
    height=120,
    key="eval_feedback_comment"
)

# -----------------------------
# Traditional search comparison
# -----------------------------

st.markdown("---")
st.header("Traditional Search Engine Comparison")

trad_found = st.radio(
    "Did you find the required information?",
    ["Yes", "No"],
    index=None,
    key="eval_trad_found"
)

trad_steps = st.number_input(
    "Number of steps taken",
    min_value=0,
    step=1,
    key="eval_trad_steps"
)

st.markdown("#### Time Taken")

col1, col2 = st.columns(2)

with col1:
    trad_minutes = st.number_input(
        "Minutes",
        min_value=0,
        step=1,
        key="eval_trad_minutes"
    )

with col2:
    trad_seconds = st.number_input(
        "Seconds",
        min_value=0,
        max_value=59,
        step=1,
        key="eval_trad_seconds"
    )

trad_total_seconds = (trad_minutes * 60) + trad_seconds


# -----------------------------
# Submit feedback
# -----------------------------

st.markdown("---")

if st.button("Submit feedback"):

    if not user_name.strip():
        st.warning("Please enter Participant Name.")

    elif not task_id.strip():

    elif found_info is None:
        st.warning("Please indicate whether you found the information using the tool.")

    elif trad_found is None:
        st.warning("Please indicate whether you found the information using traditional search.")

    else:
        feedback_row = {
            "timestamp": datetime.now().isoformat(),
            "user_type": user_type,
            "task_id": task_id,
            "user": user_name,
            "kg_time_seconds": tool_total_seconds,
            "kg_found": 1 if found_info == "Yes" else 0,
            "kg_steps": steps_taken,
            "comments": feedback_comment,
            "trad_found": 1 if trad_found == "Yes" else 0,
            "trad_steps": trad_steps,
            "trad_time_seconds": trad_total_seconds
        }

        try:
            sheet = connect_to_gsheet()
            sheet.append_row(list(feedback_row.values()))

            st.success("Feedback recorded.")

            with st.expander("Submitted record", expanded=False):
                st.dataframe(
                    pd.DataFrame([feedback_row]),
                    width="stretch"
                )

        except Exception as e:
