import json
import time

import nltk
import requests
import streamlit as st

nltk.download("punkt_tab", quiet=True)

BACKEND = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Page config & styles
# ---------------------------------------------------------------------------

st.set_page_config(page_title="ClaimChecker", layout="wide")

with open("frontend/style.css") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)


def _await_ready():
    """Block until GET /ready returns, retrying on connection errors."""
    while True:
        try:
            requests.get(f"{BACKEND}/ready")
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(1)


# ---------------------------------------------------------------------------
# Layout: main column (left) + knowledge-base panel (right)
# ---------------------------------------------------------------------------

col_main, col_upload = st.columns([3, 1], gap="large")

# ---- Right panel: Knowledge Base ----------------------------------------
with col_upload:
    st.markdown(
        "<div class='section-heading'>Knowledge Base</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-family:var(--mono);font-size:0.75rem;letter-spacing:0.08em;"
        "text-transform:uppercase;color:rgba(255,255,255,0.5);margin-bottom:0.3rem'>"
        "Click to upload &nbsp;·&nbsp; PDF &nbsp;·&nbsp; max 200 MB</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "pdf",
        type="pdf",
        label_visibility="collapsed",
        key="pdf_uploader",
    )

    if st.button(
        "Upload PDF",
        use_container_width=True,
        disabled=uploaded is None,
    ) and uploaded:
        resp = requests.post(
            f"{BACKEND}/kb/upload",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
        )
        if resp.ok:
            st.success(f"Stored {resp.json()['passages_stored']} passages.")
        else:
            st.error(f"Upload failed: {resp.text}")

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-heading' style='margin-top:0.25rem'>Indexed Documents</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Connecting..."):
        _await_ready()

    docs_resp = requests.get(f"{BACKEND}/kb/docs")
    if docs_resp.ok:
        docs = docs_resp.json()
        if docs:
            for doc in docs:
                col_title, col_btn = st.columns([5, 1])
                col_title.markdown(
                    f"<span class='small' style='color:rgba(255,255,255,0.7)'>"
                    f"{doc['doc_title']}</span>",
                    unsafe_allow_html=True,
                )
                if col_btn.button("×", key=f"del_{doc['doc_id']}", help="Delete"):
                    del_resp = requests.delete(
                        f"{BACKEND}/kb/delete", params={"doc_id": doc["doc_id"]}
                    )
                    if del_resp.ok:
                        st.success(f"Deleted '{del_resp.json()['deleted_file']}'.")
                        st.rerun()
                    else:
                        st.error(f"Delete failed: {del_resp.text}")
        else:
            st.markdown(
                "<span class='small'>No documents indexed.</span>",
                unsafe_allow_html=True,
            )
    else:
        st.error("Could not reach backend.")

# ---- Left panel: Main ----------------------------------------------------
with col_main:
    st.markdown(
        "<span class='page-title'>ClaimChecker</span>"
        "<span class='page-subtitle'>Verify factual claims against your knowledge base</span>"
        "<span class='page-subtitle'>Classify → Decompose → Verify</span>",
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Text to check",
        height=180,
        placeholder="Paste text here...",
        label_visibility="collapsed",
    )

    # Run + Stop buttons
    col_run, col_stop, _ = st.columns([1, 1, 4])
    run_clicked = col_run.button("Run", type="primary", use_container_width=True)

    stop_clicked = col_stop.button("Stop", use_container_width=True)

    if stop_clicked and "session_id" in st.session_state:
        requests.post(
            f"{BACKEND}/api/claims/pipeline/abort",
            json={"session_id": st.session_state["session_id"]},
        )
        st.session_state.pop("session_id", None)
        st.info("Pipeline aborted.")

    _STATUS_LABELS = {
        "classifying": "Classifying sentences",
        "extracting_propositions": "Extracting atomic propositions",
        "verifying": "Verifying against knowledge base",
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
            status_box.markdown(
                "<div style='font-family:var(--mono);font-size:0.7rem;color:rgba(255,255,255,0.5);'>"
                "<span class='status-indicator'></span>STARTING PIPELINE...</div>",
                unsafe_allow_html=True,
            )

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
                                label = _STATUS_LABELS.get(data_str, data_str).upper()
                                status_box.markdown(
                                    f"<div style='font-family:var(--mono);font-size:0.7rem;"
                                    f"color:rgba(255,255,255,0.5);'>"
                                    f"<span class='status-indicator'></span>{label}...</div>",
                                    unsafe_allow_html=True,
                                )
                            elif event == "result":
                                propositions = json.loads(data_str)
                            elif event == "error":
                                status_box.error(f"Pipeline stopped: {data_str}")
                                break
                            event = None

            status_box.empty()

            if propositions:
                html_parts = [
                    "<div style='margin-top:1.5rem;border-top:2px solid #fff;padding-top:1rem;'>"
                    "<div class='section-heading' style='margin-bottom:0.75rem'>Results</div>"
                ]
                prev_sentence = None
                for p in propositions:
                    verdict = p.get("verdict", "Unknown")
                    verdict_class = (
                        "verdict-verified" if verdict == "Verified"
                        else "verdict-false" if verdict in ("False", "Unverified")
                        else "verdict-unknown"
                    )
                    # Show sentence header only when it changes
                    sentence_header = ""
                    if p["sentence"] != prev_sentence:
                        sentence_label = p["sentence"][:120] + ("…" if len(p["sentence"]) > 120 else "")
                        sentence_header = (
                            f"<div class='result-sentence' style='margin-top:0.75rem'>"
                            f"{sentence_label}</div>"
                        )
                        prev_sentence = p["sentence"]

                    html_parts.append(
                        f"<div class='result-block'>"
                        f"{sentence_header}"
                        f"<div class='result-row'>"
                        f"<span class='result-claim'>{p['text']}</span>"
                        f"<span class='result-verdict {verdict_class}'>{verdict}</span>"
                        f"</div>"
                        f"</div>"
                    )
                html_parts.append("</div>")
                output_box.markdown("".join(html_parts), unsafe_allow_html=True)
            else:
                output_box.info("No factual propositions found.")
