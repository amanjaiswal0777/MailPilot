"""
Streamlit front-end for the Gmail AI agent (drafts only - it never sends mail).

Run with:  streamlit run app.py
"""
import json
import os
import secrets
import threading
import time

import streamlit as st

import gmail_agent as ga

st.set_page_config(page_title="Gmail AI Agent", page_icon="📧", layout="wide")

PAGES = ["📬 Summary", "🌅 Briefing", "💬 Chat", "📝 Drafts and invites"]

CATEGORY_LABEL = {
    "urgent": "🔴 Urgent",
    "needs_reply": "🟠 Needs reply",
    "fyi": "🔵 FYI",
    "low_priority": "⚪ Low priority",
}
URGENCY_LABEL = {"urgent": "🔴 Urgent", "normal": "🟡 Normal", "low": "⚪ Low"}

EXAMPLE_PROMPTS = [
    "Summarize my latest unread email and draft a reply",
    "Which important mails are unread?",
    "Find mails I sent that nobody replied to",
    "Which emails are waiting for my reply?",
]


# --------------------------------------------------------------------------
# State and settings
# --------------------------------------------------------------------------
def init_state():
    defaults = {
        "conn": None,  # dict: gmail, calendar, email, write_mode, calendar_enabled
        "summaries": None,
        "briefing": None,
        "agent": None,
        "chat_display": [],  # [{"role", "content", "tools"}]
        "autoconnect_tried": False,
        "hosted_oauth_url": None,
        "oauth_notice": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets.toml
        return None


def reset_outputs():
    st.session_state.summaries = None
    st.session_state.briefing = None
    reset_chat()


def reset_chat():
    st.session_state.agent = None
    st.session_state.chat_display = []


def connect(write_mode, calendar_enabled):
    """Sign in to Google. The first time, a browser window opens on the machine running Streamlit."""
    try:
        with st.spinner("Signing in to Google. A browser window may open..."):
            gmail, calendar = ga.get_services(write_mode=write_mode, calendar_enabled=calendar_enabled)
            email = ga.get_profile_email(gmail)
    except ga.ConfigError as exc:
        st.error(str(exc))
        return False
    except Exception as exc:
        st.error(f"Google sign-in failed: {exc}")
        return False

    st.session_state.conn = {
        "gmail": gmail,
        "calendar": calendar,
        "email": email,
        "write_mode": write_mode,
        "calendar_enabled": calendar_enabled,
    }
    reset_outputs()
    return True


@st.cache_resource
def _oauth_request_store():
    # Streamlit sessions are not guaranteed to survive Google's redirect.
    # Keep only short-lived, single-use CSRF state on the server.
    return {"requests": {}, "lock": threading.Lock()}


def start_hosted_oauth(write_mode, calendar_enabled):
    state = secrets.token_urlsafe(32)
    scopes = ga.scopes_for(write_mode, calendar_enabled)
    try:
        url, code_verifier = ga.create_hosted_oauth_url(scopes, state)
    except ga.ConfigError as exc:
        st.error(str(exc))
        return

    store = _oauth_request_store()
    now = time.time()
    with store["lock"]:
        store["requests"] = {
            key: value for key, value in store["requests"].items()
            if now - value["created"] < 600
        }
        store["requests"][state] = {
            "created": now,
            "scopes": scopes,
            "write_mode": write_mode,
            "calendar_enabled": calendar_enabled,
            "code_verifier": code_verifier,
        }
    st.session_state.hosted_oauth_url = url


def handle_hosted_oauth_callback():
    """Validate and finish Google's redirect before rendering the app."""
    query = st.query_params
    code, state, error = query.get("code"), query.get("state"), query.get("error")
    if not (code or error):
        return

    try:
        if error:
            raise ga.ConfigError(f"Google sign-in was not completed: {error}")
        if not code or not state:
            raise ga.ConfigError("Google returned an incomplete sign-in response. Please reconnect.")

        store = _oauth_request_store()
        with store["lock"]:
            request = store["requests"].pop(state, None)
        if not request or time.time() - request["created"] >= 600:
            raise ga.ConfigError("This sign-in request expired. Click Connect to Google and try again.")

        creds = ga.complete_hosted_oauth(
            code, state, request["scopes"], request["code_verifier"]
        )
        gmail, calendar = ga.get_services_with_credentials(
            creds, calendar_enabled=request["calendar_enabled"]
        )
        email = ga.get_profile_email(gmail)
        st.session_state.conn = {
            "gmail": gmail,
            "calendar": calendar,
            "email": email,
            "write_mode": request["write_mode"],
            "calendar_enabled": request["calendar_enabled"],
        }
        st.session_state.oauth_notice = f"Connected as {email}."
        reset_outputs()
    except ga.ConfigError as exc:
        st.session_state.oauth_notice = str(exc)
    except Exception as exc:
        st.session_state.oauth_notice = f"Google sign-in failed: {exc}"
    finally:
        # Remove the one-time authorization code from the visible URL.
        query.clear()
        st.session_state.hosted_oauth_url = None


def get_llm():
    return ga.build_llm(st.session_state.model)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.title("📧 Gmail AI Agent")
        st.caption("Reads and summarizes your mail and writes drafts. It can never send.")

        page = st.radio("Page", PAGES, label_visibility="collapsed")
        st.divider()

        conn = st.session_state.conn
        with st.expander("Settings", expanded=conn is None):
            if not os.getenv("OPENAI_API_KEY") and _secret("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = _secret("OPENAI_API_KEY")
            if not os.getenv("OPENAI_API_KEY"):
                key = st.text_input("OpenAI API key", type="password", key="openai_key")
                if key:
                    os.environ["OPENAI_API_KEY"] = key
            else:
                st.caption("OpenAI API key: loaded")

            st.text_input("Model", value=ga.DEFAULT_MODEL, key="model")
            st.text_input(
                "Your name",
                value=ga.DEFAULT_USER_NAME,
                key="user_name",
                help="Used to sign drafts. Applies to new chats.",
            )
            calendar_enabled = st.toggle(
                "Read my Google Calendar",
                value=os.getenv("ENABLE_CALENDAR", "").strip().lower() == "true",
                key="calendar_enabled",
                help="Read-only. Adds today's events to the briefing and lets the chat check availability.",
            )
            write_mode = st.toggle(
                "Save drafts to Gmail and set the Important flag",
                value=False,
                key="write_mode",
            )
            if write_mode:
                st.warning(
                    "Google has no drafts-only permission. This needs a scope that could technically "
                    "send mail. The code never calls send, but the permission exists."
                )

        if conn:
            st.success(f"Connected as {conn['email']}")
            mode = "Gmail write" if conn["write_mode"] else "Read-only"
            st.caption(f"Mode: {mode}" + (" + calendar" if conn["calendar_enabled"] else ""))
            if (conn["write_mode"], conn["calendar_enabled"]) != (write_mode, calendar_enabled):
                st.info("Permissions changed. Reconnect to apply them.")
            left, right = st.columns(2)
            if left.button("Reconnect"):
                if connect(write_mode, calendar_enabled):
                    st.rerun()
            if right.button("Disconnect"):
                st.session_state.conn = None
                reset_outputs()
                st.rerun()
        else:
            if st.button("Connect to Google", type="primary"):
                if ga.hosted_oauth_enabled():
                    start_hosted_oauth(write_mode, calendar_enabled)
                elif connect(write_mode, calendar_enabled):
                    st.rerun()
            if st.session_state.hosted_oauth_url:
                st.link_button("Continue to Google", st.session_state.hosted_oauth_url)
            # Skip the browser step when a saved login already exists.
            if not st.session_state.autoconnect_tried:
                st.session_state.autoconnect_tried = True
                if ga.token_exists(write_mode, calendar_enabled) and connect(write_mode, calendar_enabled):
                    st.rerun()
    return page


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def render_welcome():
    st.header("Your inbox, summarized")
    st.write(
        "Connect your Google account to summarize emails, get a morning briefing, "
        "and chat with an assistant that drafts replies for you to review and send yourself."
    )
    if not ga.has_oauth_client():
        st.warning(
            f"Google OAuth client file was not found at `{ga.CREDENTIALS_FILE}`. "
            "Add it locally or configure it as a Render Secret File."
        )
    if ga.hosted_oauth_enabled():
        st.info("Click **Connect to Google** in the sidebar, then approve access in Google.")
    else:
        st.info("Click **Connect to Google** in the sidebar. The first sign-in opens a browser window.")


def summaries_to_text(items):
    blocks = []
    for i, item in enumerate(items, start=1):
        e, s = item["email"], item["summary"]
        blocks.append(
            f"{i}. [{s.category}] {e['subject']}\n"
            f"   From: {e['from']}\n"
            f"   Date: {e['date']}\n"
            f"   {s.summary}\n"
            f"   Action: {s.action_item}"
        )
    return "\n\n".join(blocks)


def render_summary(conn):
    st.header("Email summary")
    st.caption("Summarizes the newest emails that match a Gmail search.")

    with st.form("summary_form"):
        query = st.text_input(
            "Gmail search",
            value="in:inbox",
            help="Any Gmail search, for example from:arpit, is:unread or newer_than:2d.",
        )
        count = st.slider("Number of emails", min_value=1, max_value=50, value=1)
        submitted = st.form_submit_button("Summarize", type="primary")

    if submitted:
        try:
            llm = get_llm()
            with st.spinner("Reading and summarizing..."):
                st.session_state.summaries = ga.summarize_query(conn["gmail"], llm, query.strip(), count)
        except ga.ConfigError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not summarize: {exc}")

    items = st.session_state.summaries
    if items is None:
        return
    if not items:
        st.info("No emails matched that search. Try a different Gmail query.")
        return

    for item in items:
        e, s = item["email"], item["summary"]
        with st.container(border=True):
            st.markdown(f"**{e['subject']}**")
            st.caption(f"{CATEGORY_LABEL[s.category]}  |  From {e['from']}  |  {e['date']}")
            st.write(s.summary)
            st.markdown(f"**Action:** {s.action_item}")

    st.download_button(
        "Download summaries",
        data=summaries_to_text(items),
        file_name="email_summary.txt",
        mime="text/plain",
    )


def render_briefing(conn):
    st.header("Morning briefing")
    st.caption("Read-only. Checks what needs your reply, what is important, and what you are waiting on.")
    if conn["calendar"] is None:
        st.info("Calendar is off, so today's events are skipped. Turn on Read my Google Calendar in Settings, then reconnect.")

    max_results = st.slider("Emails per section", min_value=5, max_value=25, value=25, step=5)

    if st.button("Generate briefing", type="primary"):
        try:
            llm = get_llm()
            with st.spinner("Going through your inbox..."):
                st.session_state.briefing = ga.generate_briefing(conn["gmail"], conn["calendar"], llm, max_results)
        except ga.ConfigError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Could not build the briefing: {exc}")

    data = st.session_state.briefing
    if not data:
        return

    st.subheader(f"{data['generated_at']:%A, %d %B %Y}")
    cols = st.columns(4)
    cols[0].metric("Need your action", len(data["pending"]))
    cols[1].metric("Unread important", len(data["important"]))
    cols[2].metric("Waiting on replies", len(data["unanswered"]))
    cols[3].metric("Events today", len(data["events"]) if data["events"] is not None else "n/a")

    st.subheader("Mails pending your attention")
    if not data["pending"] and not data["problems"]:
        st.write("Nothing pending.")
    for p in data["pending"]:
        email, a = p["email"], p["assessment"]
        with st.container(border=True):
            st.markdown(f"**{email['subject']}**")
            st.caption(f"{URGENCY_LABEL[a.urgency]}  |  From {email['from']}  |  {email['date']}")
            st.write(a.summary.strip())
            st.markdown(f"**Action needed:** {a.action.strip()}")
    for problem in data["problems"]:
        st.warning(problem)

    st.subheader("Unread important mails")
    if not data["important"]:
        st.write("Nothing pending.")
    for m in data["important"]:
        reason = " ".join(m.get("snippet", "").split()) or "No preview available"
        with st.container(border=True):
            st.markdown(f"**{m['subject']}**")
            st.caption(f"From {m['from']}")
            st.write(reason)

    st.subheader("Sent in the last 7 days with no reply")
    if not data["unanswered"]:
        st.write("Nothing pending.")
    else:
        st.dataframe(
            [
                {"Recipient": m["to"], "Subject": m["subject"], "Days waiting": m["days_waiting"]}
                for m in data["unanswered"]
            ],
            hide_index=True,
        )

    st.subheader("Today's calendar")
    if data["calendar_error"]:
        st.warning(f"Could not read your calendar: {data['calendar_error']}")
    elif data["events"] is None:
        st.write("Calendar access is off.")
    elif not data["events"]:
        st.write("Nothing pending.")
    else:
        st.dataframe(
            [{"Time": ev["time"], "Event": ev["title"], "Location": ev["location"]} for ev in data["events"]],
            hide_index=True,
        )


def _queue_prompt(text):
    st.session_state.queued_prompt = text


def _format_tool_call(name, args):
    return f"`{name}` {json.dumps(args, ensure_ascii=False)}"


def render_chat(conn):
    head, action = st.columns([5, 1])
    head.header("Chat with your inbox")
    if action.button("New chat"):
        reset_chat()
        st.rerun()
    st.caption("The assistant remembers earlier messages in this chat. Model and name changes apply to new chats.")

    prompt = st.chat_input("Ask about your mail, or tell me to draft a reply")
    prompt = prompt or st.session_state.pop("queued_prompt", None)

    if not st.session_state.chat_display and not prompt:
        st.write("Try one of these:")
        for text in EXAMPLE_PROMPTS:
            st.button(text, on_click=_queue_prompt, args=(text,), key=f"example_{text}")

    for msg in st.session_state.chat_display:
        with st.chat_message(msg["role"]):
            if msg.get("tools"):
                with st.expander(f"{len(msg['tools'])} tool call(s)"):
                    for line in msg["tools"]:
                        st.markdown(line)
            st.markdown(msg["content"])

    if not prompt:
        return

    st.session_state.chat_display.append({"role": "user", "content": prompt, "tools": []})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status = st.status("Working on it...", expanded=True)
        tool_log = []

        def on_tool_call(name, args):
            line = _format_tool_call(name, args)
            tool_log.append(line)
            status.write(line)

        try:
            if st.session_state.agent is None:
                st.session_state.agent = ga.build_agent(
                    conn["gmail"],
                    get_llm(),
                    calendar=conn["calendar"],
                    write_mode=conn["write_mode"],
                    user_name=st.session_state.user_name,
                )
            answer = st.session_state.agent.ask(prompt, on_tool_call=on_tool_call)
            status.update(label="Done", state="complete", expanded=False)
        except ga.ConfigError as exc:
            status.update(label="Missing setup", state="error")
            answer = f"⚠️ {exc}"
        except Exception as exc:
            status.update(label="Something went wrong", state="error")
            answer = f"⚠️ {exc}"
        st.markdown(answer)

    st.session_state.chat_display.append({"role": "assistant", "content": answer, "tools": tool_log})


def render_drafts():
    st.header("Drafts and invites")
    st.caption(
        "Everything the assistant prepares is saved here on your computer. Nothing is sent. "
        "Open a .eml file in your mail app to edit and send it; import a .ics file into Google Calendar."
    )

    st.subheader("Drafts")
    drafts = ga.list_saved_drafts()
    if not drafts:
        st.write("No drafts yet. Ask the assistant in Chat to draft a reply.")
    for d in drafts:
        with st.expander(f"{d['subject']}  |  to {d['to']}  |  {d['saved_at']:%d %b %H:%M}"):
            st.text(d["body"])
            left, right, _ = st.columns([1, 1, 4])
            left.download_button(
                "Download .txt",
                data=d["txt_path"].read_bytes(),
                file_name=d["txt_path"].name,
                mime="text/plain",
                key=f"txt_{d['stem']}",
            )
            if d["eml_path"]:
                right.download_button(
                    "Download .eml",
                    data=d["eml_path"].read_bytes(),
                    file_name=d["eml_path"].name,
                    mime="message/rfc822",
                    key=f"eml_{d['stem']}",
                )

    st.subheader("Calendar invites")
    invites = ga.list_saved_invites()
    if not invites:
        st.write("No invite files yet. Ask the assistant in Chat to prepare a call invite.")
    for inv in invites:
        with st.expander(inv["name"]):
            st.code(inv["text"], language="text")
            st.download_button(
                "Download .ics",
                data=inv["path"].read_bytes(),
                file_name=inv["name"],
                mime="text/calendar",
                key=f"ics_{inv['name']}",
            )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    init_state()
    if ga.hosted_oauth_enabled():
        handle_hosted_oauth_callback()
    if st.session_state.oauth_notice:
        if st.session_state.conn:
            st.success(st.session_state.oauth_notice)
        else:
            st.error(st.session_state.oauth_notice)
        st.session_state.oauth_notice = None
    page = render_sidebar()
    conn = st.session_state.conn

    if page == PAGES[3]:  # drafts are local files, so no sign-in is needed to view them
        render_drafts()
    elif not conn:
        render_welcome()
    elif page == PAGES[0]:
        render_summary(conn)
    elif page == PAGES[1]:
        render_briefing(conn)
    else:
        render_chat(conn)


main()
