<<<<<<< HEAD
import streamlit as st
import pandas as pd
from datetime import datetime
import gspread

from google.oauth2.service_account import Credentials
from utils import show_consent_dialog

st.set_page_config(
    page_title="Feedback",
    layout="wide"
)

st.title("Feedback")

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

# -----------------------------
# Tool feedback
# -----------------------------
st.markdown("---")
st.header("Tool Feedback")

feedback_comment = st.text_area(
    "Comments",
    placeholder="Comments",
    height=120,
    key="eval_feedback_comment"
)


# -----------------------------
# Submit feedback
# -----------------------------

st.markdown("---")

if st.button("Submit feedback"):

    if not user_name.strip():
        st.warning("Please enter Participant Name.")

    else:
        feedback_row = {
            "timestamp": datetime.now().isoformat(),
            "user_type": user_type,
            "task_id": "10",
            "user": user_name,
            "kg_time_seconds": "n/a",
            "kg_found": "n/a",
            "kg_steps": "n/a",
            "comments": feedback_comment,
            "trad_found": "n/a",
            "trad_steps": "n/a",
            "trad_time_seconds": "n/a"
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
            st.error(f"Feedback could not be recorded: {e}")

=======
import streamlit as st
import pandas as pd
from datetime import datetime
import gspread

from google.oauth2.service_account import Credentials


st.set_page_config(
    page_title="Feedback",
    layout="wide"
)

st.title("Feedback")

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

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

# -----------------------------
# Tool feedback
# -----------------------------
st.markdown("---")
st.header("Tool Feedback")

feedback_comment = st.text_area(
    "Comments",
    placeholder="Comments",
    height=120,
    key="eval_feedback_comment"
)


# -----------------------------
# Submit feedback
# -----------------------------

st.markdown("---")

if st.button("Submit feedback"):

    if not user_name.strip():
        st.warning("Please enter Participant Name.")

    else:
        feedback_row = {
            "timestamp": datetime.now().isoformat(),
            "user_type": user_type,
            "task_id": "10",
            "user": user_name,
            "kg_time_seconds": "n/a",
            "kg_found": "n/a",
            "kg_steps": "n/a",
            "comments": feedback_comment,
            "trad_found": "n/a",
            "trad_steps": "n/a",
            "trad_time_seconds": "n/a"
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
            st.error(f"Feedback could not be recorded: {e}")

>>>>>>> 45208bfb7eee502018090845c66f1f7d6ee59730
