import json
import time

import nltk
import requests
import streamlit as st

nltk.download("punkt_tab", quiet=True)

BACKEND = "http://localhost:8000"


def _await_ready():
    """Block until GET /ready returns, retrying on connection errors."""
    while True:
        try:
            requests.get(f"{BACKEND}/ready")
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(1)

st.title("ClaimChecker")

# ---------------------------------------------------------------------------
# KB sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Knowledge Base")

    uploaded = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Upload") and uploaded:
        resp = requests.post(
            f"{BACKEND}/kb/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
        )
        if resp.ok:
            st.success(f"Stored {resp.json()['passages_stored']} passages.")
        else:
            st.error(f"Upload failed: {resp.text}")

    st.subheader("Indexed Documents")
    with st.spinner("Waiting for backend…"):
        _await_ready()
    docs_resp = requests.get(f"{BACKEND}/kb/docs")
    if docs_resp.ok:
        docs = docs_resp.json()
        if docs:
            for doc in docs:
                col_title, col_btn = st.columns([5, 1])
                col_title.markdown(f"**{doc['doc_title']}** (`{doc['doc_id'][:8]}…`)")
                if col_btn.button("🗑", key=f"del_{doc['doc_id']}", help="Delete document"):
                    del_resp = requests.delete(
                        f"{BACKEND}/kb/delete", params={"doc_id": doc["doc_id"]}
                    )
                    if del_resp.ok:
                        st.success(f"Deleted '{del_resp.json()['deleted_file']}'.")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {del_resp.text}")
        else:
            st.caption("No documents uploaded yet.")
    else:
        st.error("Could not reach backend.")

# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

text = st.text_area("Text to check", height=200, placeholder="Paste text here…")

col_run, col_stop = st.columns([1, 1])
run_clicked = col_run.button("Run Pipeline", type="primary")
stop_clicked = col_stop.button("Stop")

if stop_clicked and "session_id" in st.session_state:
    requests.post(
        f"{BACKEND}/api/claims/pipeline/abort",
        json={"session_id": st.session_state["session_id"]},
    )
    st.session_state.pop("session_id", None)
    st.info("Pipeline aborted.")

_STATUS_LABELS = {
    "classifying": "Classifying sentences (fact vs opinion)…",
    "extracting_propositions": "Extracting atomic propositions…",
    "verifying": "Verifying propositions against knowledge base…",
}

status_box = st.empty()
output_box = st.empty()

if run_clicked:
    if not text.strip():
        st.warning("Enter some text first.")
    else:
        sentences = nltk.sent_tokenize(text)
        st.session_state.pop("session_id", None)

        propositions: list[dict] = []
        status_box.info("Starting pipeline…")

        with requests.post(
            f"{BACKEND}/api/claims/pipeline/run",
            json={"sentences": sentences},
            stream=True,
            timeout=300,
        ) as resp:
            if not resp.ok:
                status_box.error(f"Pipeline error: {resp.text}")
            else:
                event = None
                for raw in resp.iter_lines():
                    if not raw:
                        continue
                    line = raw.decode() if isinstance(raw, bytes) else raw

                    if line.startswith("event:"):
                        event = line.removeprefix("event:").strip()
                    elif line.startswith("data:") and event is not None:
                        data_str = line.removeprefix("data:").strip()

                        if event == "session":
                            st.session_state["session_id"] = data_str
                        elif event == "status":
                            label = _STATUS_LABELS.get(data_str, data_str)
                            status_box.info(label)
                        elif event == "result":
                            propositions = json.loads(data_str)
                        elif event == "error":
                            status_box.error(f"Pipeline stopped: {data_str}")
                            break
                        event = None

        status_box.empty()

        if propositions:
            lines = []
            for p in propositions:
                verdict = p.get("verdict", "Unknown")
                icon = "✅" if verdict == "Verified" else "❌"
                lines.append(f"**Sentence:** {p['sentence']}")
                lines.append(f"- {p['text']} — {icon} {verdict}")
                lines.append("")
            output_box.markdown("\n".join(lines))
        else:
            output_box.info("No factual propositions found.")
