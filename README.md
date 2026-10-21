# Gmail AI Agent (Streamlit)

A Streamlit app that summarizes your email, builds a morning briefing, and chats with an
assistant that drafts replies. **It never sends mail**: drafts are saved for you to review and send yourself.

## Setup

1. **Google Cloud**: create a project, enable the **Gmail API** (and the **Calendar API** if you want calendar),
   create an OAuth client ID, and add your Google account as a test user while the consent screen is in "Testing".
   For a local run, choose **Desktop app** and save the download here as `credentials.json`. For Render, use a
   separate **Web application** OAuth client; see the hosted setup below.
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

## Deploy on Render

The hosted app uses Google's web-server OAuth flow. A Desktop app OAuth file cannot complete sign-in on Render.

1. In Google Cloud Console, create a separate OAuth client ID with application type **Web application**.
2. Add your exact Render URL with a trailing slash as an authorized redirect URI, for example
   `https://your-service.onrender.com/`.
3. In the Render service, open **Environment**. Add `GOOGLE_REDIRECT_URI` with that same URL and add
   `OPENAI_API_KEY` with your OpenAI key.
4. In **Secret Files**, add a file named `credentials.json` and paste the downloaded Web application OAuth JSON
   into it. Render makes it available at `/etc/secrets/credentials.json`; do not commit it to GitHub.
5. Save the changes and redeploy. Open the app, click **Connect to Google**, then approve access in Google.

Keep the OAuth consent screen in Testing and add the Google account you use to sign in as a test user. The hosted
login keeps its access token in the current app session, so you may need to connect again after the session ends.

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

- For hosted deployment, configure the Web application OAuth client and redirect URI as described above.
- Keep `credentials.json`, `token_*.json` and `.env` private (they are in `.gitignore`). To sign out for good, delete the `token_*.json` files.
- Email text is treated as untrusted data, and draft recipients come from the original email, not from the model.
