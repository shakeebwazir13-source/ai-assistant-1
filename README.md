# ai-assistant-1
# 📄 ATS Resume Analyzer

A Streamlit app that scores a resume for Applicant Tracking Systems (ATS) and suggests concrete improvements, powered by Google Gemini Flash.

## Features
- Upload a resume as **PDF, DOCX or TXT**
- Overall **ATS score (0-100)** with a breakdown: formatting, keywords, content quality, structure, impact
- Optional **job description** for tailored keyword matching
- Strengths, prioritized improvements, missing keywords, before/after bullet rewrites
- Download the report as JSON

## Tech stack
Python, Streamlit, Google Gemini (`gemini-2.5-flash`) via the `google-genai` SDK, `pypdf`, `python-docx`

## Run locally
```bash
git clone https://github.com/<your-username>/ats-resume-analyzer.git
cd ats-resume-analyzer
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Get a free API key at https://aistudio.google.com/apikey, then provide it in one of these ways:

1. Create `.streamlit/secrets.toml` (never commit this file):
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
2. Or set an environment variable (PowerShell): `$env:GEMINI_API_KEY="your-key-here"`
3. Or paste it into the app's sidebar when it asks.

Start the app:
```bash
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, pick the repo, branch `main`, main file `app.py`.
4. Open **Advanced settings → Secrets** and add: `GEMINI_API_KEY = "your-key-here"`
5. Click **Deploy**.

## Optional configuration
- `GEMINI_MODEL` environment variable overrides the model name (default `gemini-2.5-flash`).

## Notes
- Scanned/image-only PDFs have no readable text; use a text-based PDF or DOCX.
- The ATS score is an AI estimate, not the output of a real ATS.
- Never commit your API key. Add `.streamlit/secrets.toml` and `venv/` to `.gitignore`.
