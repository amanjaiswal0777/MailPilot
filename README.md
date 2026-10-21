# Gmail AI Agent (Streamlit)

A Streamlit app that summarizes your email, builds a morning briefing, and chats with an
assistant that drafts replies. **It never sends mail**: drafts are saved for you to review and send yourself.

## Setup

1. **Google Cloud**: create a project, enable the **Gmail API** (and the **Calendar API** if you want calendar),
   create an OAuth client ID of type **Desktop app**, download it and save it here as `credentials.json`.
   While the app is in "Testing", add your Google account as a test user.
2. **Install**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Configure**: copy `.env.example` to `.env` and set `OPENAI_API_KEY` (you can also paste the key in the sidebar).
4. **Run**
   ```bash
   streamlit run app.py
   ```
   Click **Connect to Google** in the sidebar. The first sign-in opens a browser window; after that the saved
   `token_*.json` is reused automatically.

## Pages

| Page | Replaces CLI command |
|---|---|
| Summary | `summary --query ... --max N` |
| Briefing | `briefing` |
| Chat | `ask ...` and `chat` (keeps context; use **New chat** to reset) |
| Drafts and invites | Browse and download saved `.eml`, `.txt` and `.ics` files |

## Permissions

- Default is **read-only** (`gmail.readonly`). Drafts go to the local `drafts/` folder, "important" marks to `important_marks.json`.
- The sidebar toggle **Save drafts to Gmail and set the Important flag** switches to `gmail.modify`. Google has no
  drafts-only scope, so that permission could technically send mail. The code never calls send.
- **Read my Google Calendar** adds `calendar.readonly`.
- After changing a toggle, click **Reconnect**. Each permission set has its own token file.

## Files

- `app.py`: Streamlit UI
- `gmail_agent.py`: Gmail, Calendar, summary chain, briefing and agent logic (no printing, so the UI can reuse it)

## Notes

- Run this **on your own computer**. Google sign-in opens a browser on the machine running Streamlit, so it will
  not work as-is on a hosted server.
- Keep `credentials.json`, `token_*.json` and `.env` private (they are in `.gitignore`). To sign out for good, delete the `token_*.json` files.
- Email text is treated as untrusted data, and draft recipients come from the original email, not from the model.
