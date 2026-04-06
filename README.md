# Meeting Intelligence Copilot

Upload a meeting transcript or audio file, or load a public QMSum benchmark sample, then extract:

- executive summaries
- key decisions
- action items and likely owners
- risks / blockers
- transcript-grounded answers to natural-language questions

## Why this project

Most demo RAG apps stop at “chat with a PDF.” This project is a more product-like prototype focused on a common business workflow: turning raw meetings into usable insights.

## Main features

- Streamlit web app with a polished multi-tab UI
- Transcript upload support (`.txt`, `.md`, `.json`)
- Audio upload support (`.wav`, `.mp3`, `.m4a`) using Whisper
- Local retrieval pipeline with sentence-transformer embeddings
- Structured insight extraction without requiring an API key
- Optional OpenAI mode for richer summaries and answers
- Script to pull a public QMSum sample from the official repository
- Docker support for quick local deployment

## Project structure

```text
meeting-intelligence-copilot/
├── app.py
├── data/
│   ├── sample_meeting_synthetic.json
│   └── qmsum_sample.json              # created after running the download script
├── scripts/
│   └── download_qmsum_sample.py
├── utils/
│   ├── audio_tools.py
│   ├── dataset_loader.py
│   ├── llm.py
│   └── nlp.py
├── .streamlit/
│   └── config.toml
├── Dockerfile
├── docker-compose.yml
├── packages.txt
├── requirements.txt
└── README.md
```

## Input and output

### Input

The app accepts one of these sources:

1. Included synthetic transcript sample
2. Public QMSum benchmark sample downloaded locally
3. Uploaded transcript file
4. Uploaded audio file

### Output

The app shows:

- a manager-style executive summary
- extracted decisions
- extracted action items
- extracted risks / blockers
- transcript-grounded answers to free-form questions
- supporting transcript snippets for verification

## Quickstart

### 1) Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run the app

```bash
streamlit run app.py
```

## Public dataset option

To fetch one public sample from the official QMSum repository into `data/qmsum_sample.json`, run:

```bash
python scripts/download_qmsum_sample.py
```

The script clones the official repository, finds one meeting JSON, and saves it locally for the app.

## Optional OpenAI support

The app works without an API key.

If you want richer summaries and answers, set:

```bash
OPENAI_API_KEY=your_key_here
```

Or enter the key directly in the app sidebar.

## Docker

Build and run:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8501
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a new GitHub repository.
2. In Streamlit Community Cloud, choose the repo and `app.py`.
3. Add `OPENAI_API_KEY` as a secret if you want LLM mode.
4. Add system packages if needed for audio transcription support. `packages.txt` already lists `ffmpeg` and `git` for compatible hosts.

## Example questions

- What are the key decisions?
- List the action items and owners.
- What risks or blockers were mentioned?
- What timeline did the team agree on?
- What was decided about the MVP scope?

## Notes

- Audio transcription uses Whisper locally, so the first transcription run can take longer while the model downloads.
- The local extraction pipeline is intentionally conservative. It retrieves evidence and surfaces likely decisions, actions, and risks using rules and semantic search.
- The OpenAI path is optional and improves fluency, but the project remains usable without it.

## License

This project is released under the MIT License.
