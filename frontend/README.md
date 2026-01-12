Minimal UI frontend for the FastAPI backend.

## Tech Stack
- Next.js (App Router)
- TypeScript
- TailwindCSS
- Yarn

---

## Setup

Install dependencies:
```bash
yarn
```

Create environment file: `frontend/.env.local`
```bash
NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
```

## Run (Development)

Start the backend first (from repo root):

```bash
python -m uvicorn src.api:app --reload --port 8000
```

Start the frontend (from frontend/):
```bash
yarn dev
```

Open: http://localhost:3000
