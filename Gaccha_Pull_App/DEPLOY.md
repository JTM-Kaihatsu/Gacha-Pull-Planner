# Deployment

The app is two pieces that deploy separately:

- **Backend** (FastAPI) — a running Python process → **Render** (free tier).
- **Frontend** (React/Vite) — static files → **Vercel** (free tier).

They talk over HTTP, so each needs to know the other's URL. Deploy the backend
first, then the frontend, then point them at each other.

---

## 1. Backend → Render

1. **New + → Web Service**, connect this GitHub repo.
2. Set **Root Directory** to `Gaccha_Pull_App`.
3. Render should auto-fill from [`render.yaml`](render.yaml); otherwise set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Under **Environment**, add:
   | Key | Value |
   |-----|-------|
   | `ALLOWED_ORIGINS` | your frontend URL (fill in after step 2 below), e.g. `https://your-app.vercel.app` |
   | `OPENAI_API_KEY` | your key — **only needed if you use the AI toggle** (see below) |
   | `OPENAI_MODEL` | `gpt-4o` (optional) |
5. Deploy. Note the URL, e.g. `https://gacha-pull-simulator-api.onrender.com`.
   Confirm it's up by visiting `<url>/docs` (Swagger UI).

> Free tier spins down after ~15 min idle, so the first request after a quiet
> period takes ~30–50s to wake. A paid instance (~$7/mo) stays always-on.

## 2. Frontend → Vercel

1. **Add New → Project**, import this repo.
2. Set **Root Directory** to `Gaccha_Pull_App/frontend`. Vercel auto-detects Vite
   (build `npm run build`, output `dist`); [`vercel.json`](frontend/vercel.json)
   pins this and adds the SPA rewrite.
3. Add an **Environment Variable**:
   | Key | Value |
   |-----|-------|
   | `VITE_API_URL` | your Render backend URL from step 1 |
4. Deploy. Note the URL, e.g. `https://your-app.vercel.app`.

## 3. Wire them together

- Put the **Vercel URL** into the backend's `ALLOWED_ORIGINS` on Render (CORS), and
  redeploy the backend. `ALLOWED_ORIGINS` accepts a comma-separated list.
- Confirm the **Render URL** is in the frontend's `VITE_API_URL` on Vercel.

Visit the Vercel URL and click **Simulate** — the browser loads the static frontend,
which calls the Render backend.

---

## Cost & the AI toggle

- **Hosting** is free on both tiers (trade-off: slow first backend request when idle).
- **The AI interpretation is the only thing that costs money** (OpenAI usage on your
  key). It is **off by default** — `enable_ai_analysis` defaults to `false`, so the
  public demo runs the full simulation and chart **without any OpenAI calls**. Users
  opt in via **Advanced Settings → Enable AI interpretation**.
- If you leave `OPENAI_API_KEY` unset on Render, the app still works fully; only the
  optional AI verdict is unavailable (enabling it without a key returns a 500).
- Recommended: set a **hard monthly spend cap** in your OpenAI dashboard (e.g. $5) as
  a safety net before sharing a public link.
