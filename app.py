"""
Streamlit Web Application for Data Analysis Agent.

Allows users to upload tabular files (CSV, TSV, Excel, JSON, Parquet),
view dataset metrics and previews, and chat with an LLM-powered data agent.
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import openai

from data_loader import load_table
from agent import DataAgent


# Page configuration
st.set_page_config(
    page_title="Data Analysis Agent",
    page_icon="📊",
    layout="wide",
)


def render_dataframe(df: pd.DataFrame):
    """Render dataframe avoiding deprecated use_container_width."""
    try:
        st.dataframe(df, width="stretch")
    except TypeError:
        try:
            st.dataframe(df, use_container_width=True)
        except TypeError:
            st.dataframe(df)


def render_image(image_path: str, caption: str = None):
    """Render image avoiding deprecated use_container_width."""
    try:
        st.image(image_path, caption=caption, width="stretch")
    except TypeError:
        try:
            st.image(image_path, caption=caption, use_container_width=True)
        except TypeError:
            st.image(image_path, caption=caption)


def handle_llm_exception(e: Exception) -> str:
    """Format known LLM exceptions into clear, friendly guidance."""
    err_str = str(e)
    status_code = getattr(e, "status_code", None)

    if status_code == 401 or "401" in err_str or isinstance(e, openai.AuthenticationError):
        return (
            "Authentication Error (401): Invalid API key.\n\n"
            "Please check that your API key is correctly entered in the sidebar or configured as LLM_API_KEY in your .env file."
        )
    elif status_code == 429 or "429" in err_str or isinstance(e, openai.RateLimitError):
        return (
            "Rate Limit Exceeded (429): You have exceeded your API quota or rate limit.\n\n"
            "Please wait a brief moment before sending your next request, or check your provider quota."
        )
    elif status_code == 404 or "404" in err_str or isinstance(e, openai.NotFoundError):
        return (
            "Model Not Found (404): The requested model does not exist on this endpoint.\n\n"
            "Please verify the model name in the sidebar or your .env file."
        )
    elif isinstance(e, openai.APIConnectionError) or "connection" in err_str.lower():
        return (
            "Connection Error: Unable to reach the LLM API endpoint.\n\n"
            "Please check your network connection and ensure the Base URL is reachable."
        )
    else:
        return f"Analytical Query Error: {e}"


# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of dicts: {role, content, tool_calls, plots}

if "agent" not in st.session_state:
    st.session_state.agent = None

if "current_file_id" not in st.session_state:
    st.session_state.current_file_id = None

if "df" not in st.session_state:
    st.session_state.df = None


# --- SIDEBAR CONTROLS ---
st.sidebar.title("Data & Settings")

# 1. File uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Tabular Dataset",
    type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
    help="Upload any CSV, TSV, Excel, JSON, or Parquet file.",
)

# 2. Tool calls visibility toggle
show_tool_calls = st.sidebar.checkbox(
    "Show tool calls",
    value=True,
    help="Toggle to display code execution steps and outputs from the agent.",
)

# 3. Optional endpoint configuration overrides
st.sidebar.subheader("LLM Configuration")
override_api_key = st.sidebar.text_input(
    "API Key (optional override)",
    type="password",
    value="",
    help="Leave empty to use LLM_API_KEY from .env",
)
override_base_url = st.sidebar.text_input(
    "Base URL (optional override)",
    value="",
    placeholder="https://api.groq.com/openai/v1",
    help="Leave empty to use LLM_BASE_URL from .env",
)
override_model = st.sidebar.text_input(
    "Model (optional override)",
    value="",
    placeholder="llama-3.3-70b-versatile",
    help="Leave empty to use LLM_MODEL from .env",
)


# Handle dataset upload and agent instantiation
if uploaded_file is not None:
    # Determine unique file identifier (name + size)
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.current_file_id != file_id:
        try:
            df = load_table(uploaded_file)
            st.session_state.df = df
            st.session_state.current_file_id = file_id
            st.session_state.chat_history = []  # Reset conversation for new dataset

            # Build fresh agent with current configuration
            st.session_state.agent = DataAgent(
                df=df,
                api_key=override_api_key if override_api_key.strip() else None,
                base_url=override_base_url if override_base_url.strip() else None,
                model=override_model if override_model.strip() else None,
            )
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            st.session_state.df = None
            st.session_state.agent = None

    elif st.session_state.agent is not None:
        # Update agent parameters if overrides changed dynamically
        if override_api_key.strip():
            st.session_state.agent.api_key = override_api_key.strip()
            st.session_state.agent.client.api_key = override_api_key.strip()
        if override_base_url.strip():
            st.session_state.agent.base_url = override_base_url.strip()
            st.session_state.agent.client.base_url = override_base_url.strip()
        if override_model.strip():
            st.session_state.agent.model = override_model.strip()


# --- MAIN INTERFACE ---
st.title("Data Analysis Agent")
st.caption("Ask questions of any dataset. Every figure is computed, never guessed.")

if st.session_state.df is not None:
    df = st.session_state.df

    # Dataset Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rows", f"{len(df):,}")
    col2.metric("Total Columns", f"{len(df.columns):,}")
    col3.metric("Missing Cells", f"{df.isnull().sum().sum():,}")
    col4.metric("Duplicate Rows", f"{df.duplicated().sum():,}")

    # Dataset Preview (first 10 rows)
    with st.expander("Dataset Preview (First 10 Rows)", expanded=False):
        render_dataframe(df.head(10))

else:
    st.info("Please upload a CSV, TSV, Excel, JSON, or Parquet dataset using the sidebar to begin.")


# --- CHAT INTERFACE ---
# Render historical messages across reruns
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display tool calls if requested
        if show_tool_calls and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                tool_label = f"Tool: {tc['tool']}"
                with st.expander(tool_label, expanded=False):
                    if "code" in tc.get("arguments", {}):
                        st.code(tc["arguments"]["code"], language="python")
                    st.text(tc["result"])

        # Render generated charts inline
        if msg.get("plots"):
            for plot_path in msg["plots"]:
                if os.path.exists(plot_path):
                    render_image(plot_path)


# User question input
user_query = st.chat_input("Ask a question about your dataset...")

if user_query:
    if st.session_state.df is None or st.session_state.agent is None:
        st.warning("Please upload a dataset before asking questions.")
    else:
        # Display user message immediately
        st.chat_message("user").markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Reviewing the data..."):
                try:
                    res = st.session_state.agent.ask(user_query)
                    answer_text = res["answer"]
                    tool_calls = res.get("tool_calls", [])
                    plots = res.get("plots", [])

                    st.markdown(answer_text)

                    if show_tool_calls and tool_calls:
                        for tc in tool_calls:
                            with st.expander(f"Tool: {tc['tool']}", expanded=False):
                                if "code" in tc.get("arguments", {}):
                                    st.code(tc["arguments"]["code"], language="python")
                                st.text(tc["result"])

                    if plots:
                        for p in plots:
                            if os.path.exists(p):
                                render_image(p)

                    # Store in chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": user_query,
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer_text,
                        "tool_calls": tool_calls,
                        "plots": plots,
                    })

                except Exception as ex:
                    error_msg = handle_llm_exception(ex)
                    st.error(error_msg)
