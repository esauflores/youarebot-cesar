import os
import uuid
import requests
import streamlit as st

ORCHESTRATOR_URL = os.environ.get("ORCHESTRATOR_URL", "http://orchestrator:8672")

st.set_page_config(page_title="YouAreBot v2", layout="wide", page_icon="🤖")


def check_health():
    try:
        r = requests.get(f"{ORCHESTRATOR_URL}/health", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


with st.sidebar:
    st.title("YouAreBot v2")
    st.divider()
    if check_health():
        st.success("API connected")
    else:
        st.error("API unreachable")
    with st.expander("Endpoints", expanded=False):
        st.code(f"GET  {ORCHESTRATOR_URL}/health\nPOST {ORCHESTRATOR_URL}/get_message\nPOST {ORCHESTRATOR_URL}/predict")
    st.divider()
    st.caption("ModernBERT + DoRA · Qwen3.5 0.8B")


def init_chat():
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def reset_chat():
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []


def send_chat(user_text: str) -> str | None:
    payload = {
        "dialog_id": st.session_state.chat_id,
        "last_msg_text": user_text,
        "last_message_id": str(uuid.uuid4()),
    }
    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}/get_message", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("new_msg_text")
    except requests.RequestException as e:
        st.toast(f"Error: {e}", icon="❌")
        return None


def classify(text: str) -> float | None:
    try:
        resp = requests.post(f"{ORCHESTRATOR_URL}/predict", json={"text": text}, timeout=10)
        resp.raise_for_status()
        return resp.json().get("is_bot_probability")
    except requests.RequestException as e:
        st.toast(f"Error: {e}", icon="❌")
        return None


init_chat()
tab_chat, tab_clf = st.tabs(["💬 Chat", "🔍 Classifier"])

with tab_chat:
    col_main, col_side = st.columns([3, 1])
    with col_side:
        st.button("🔄 New chat", on_click=reset_chat, use_container_width=True)
        st.caption(f"`{st.session_state.chat_id[:12]}...`")
    with col_main:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if prompt := st.chat_input("Say something..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    reply = send_chat(prompt)
                if reply:
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                else:
                    st.caption("*(no response)*")

with tab_clf:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Detect bot-written text")
        st.caption("Higher score = more likely AI-generated.")
    with col2:
        if "clf_history" not in st.session_state:
            st.session_state.clf_history = []
        if st.session_state.clf_history:
            st.button("🗑 Clear", on_click=lambda: st.session_state.update(clf_history=[]))

    clf_text = st.text_area("Message", placeholder="Paste a message here...", height=120, label_visibility="collapsed")

    if st.button("Analyze", type="primary", use_container_width=True):
        if not clf_text.strip():
            st.warning("Enter a message first.")
        else:
            prob = classify(clf_text.strip())
            if prob is not None:
                if prob > 0.6:
                    emoji, color, label = "🤖", "#e74c3c", "Likely bot"
                elif prob > 0.4:
                    emoji, color, label = "🤔", "#f39c12", "Uncertain"
                else:
                    emoji, color, label = "👤", "#27ae60", "Likely human"
                cols = st.columns([1, 4])
                with cols[0]:
                    st.metric("Score", f"{emoji} {prob:.0%}")
                with cols[1]:
                    st.progress(prob, text=f"{label} · {prob:.0%}")
                entry = {"text": clf_text.strip()[:80], "score": prob, "label": label}
                st.session_state.clf_history.insert(0, entry)

    if st.session_state.get("clf_history"):
        st.divider()
        st.caption("Recent")
        for h in st.session_state.clf_history[:10]:
            emoji = {"Likely bot": "🤖", "Uncertain": "🤔", "Likely human": "👤"}[h["label"]]
            st.markdown(f"{emoji} **{h['score']:.0%}** — *{h['text']}*")
