import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import streamlit as st
from pathlib import Path
from src.ingestion import ingest
from src.vectorstore import get_all_ingested_docs
from src import router

st.set_page_config(page_title="Financial Docs RAG", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = {}


st.sidebar.title("Document Management")

uploaded_file = st.sidebar.file_uploader("Upload PDF or CSV", type=["pdf", "csv"])

if uploaded_file:
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    file_path = data_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.sidebar:
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            doc_id = ingest(str(file_path))
            st.session_state.ingested_docs = get_all_ingested_docs()
            st.success(f"Ingested {uploaded_file.name}")

if not st.session_state.ingested_docs:
    st.session_state.ingested_docs = get_all_ingested_docs()

st.sidebar.subheader("Ingested Documents")
if st.session_state.ingested_docs:
    for doc_id, info in st.session_state.ingested_docs.items():
        st.sidebar.text(f"- {info['source']} ({info['chunks']} chunks)")
else:
    st.sidebar.info("No documents ingested yet")

st.sidebar.subheader("Scope")
scope_options = ["All documents"] + list(st.session_state.ingested_docs.keys())
selected_scope = st.sidebar.selectbox("Search scope:", scope_options)
doc_id_filter = None if selected_scope == "All documents" else selected_scope

st.title("Financial Docs RAG Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "intent" in message:
            st.caption(f"Intent: {message['intent']}")
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question, request a summary, or quiz yourself"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = router.handle(prompt, doc_id=doc_id_filter)
                intent = response["intent"]
                result = response["result"]

                st.caption(f"Intent: {intent}")

                if intent == "SUMMARIZE":
                    output = result.replace("$", "\\$")
                    st.markdown(output)

                elif intent == "QA":
                    output = str(result).replace("$", "\\$")
                    st.markdown(output)

                elif intent == "MCQ":
                    output = ""
                    if isinstance(result, list):
                        for i, mcq in enumerate(result, 1):
                            st.markdown(f"**{i}. {mcq.question.replace('$', '\\$')}**")
                            selected = st.radio(
                                f"Question {i}",
                                mcq.options,
                                key=f"mcq_{len(st.session_state.messages)}_{i}",
                                label_visibility="collapsed"
                            )
                            with st.expander("Show answer"):
                                correct = mcq.options[mcq.answer_index].replace("$", "\\$")
                                st.success(f"Correct answer: {correct}")
                                st.info(f"Explanation: {mcq.explanation.replace('$', '\\$')}")
                            output += f"{i}. {mcq.question}\n"
                    else:
                        output = str(result).replace("$", "\\$")
                        st.markdown(output)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": output,
                    "intent": intent
                })

            except ValueError as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
