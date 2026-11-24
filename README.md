# Ishara — systems

**Ishara** is an Android app (Flutter, sister repo `ishara-app`) that teaches **Nepali Sign Language (NSL)** with
short gamified lessons and on-device camera feedback. This repository holds the parts that don't run on the
phone:

| Folder | What it is |
|---|---|
| `backend/` | REST API (`/api/v1`): Express 5 + TypeScript + Drizzle ORM on Supabase (Postgres, Auth, Storage) |
| `ml/` | Python pipeline: MediaPipe hand landmarks from public NSL data → fingerspelling TFLite model + greeting DTW templates |
| `docs/` | API contract, data model and game rules; kept in sync with the code |

Thesis project (ST6000CEM), v1 scope.

## Status

- **Backend:** skeleton only: config validation, database connection, error handling, `GET /api/v1/health`.
  Tables, auth and feature endpoints come next.
- **ML:** environment set up (MediaPipe, TensorFlow, scikit-learn and others) with smoke tests. The pipeline
  modules come next.

## Backend quick start

Requires Node.js ≥ 24.

```bash
cd backend
npm install
npm run dev          # http://localhost:4000
```

Create `backend/.env.local`. It's git-ignored, and loaded before `.env`:

| Variable | Example / notes |
|---|---|
| `PORT` | `4000` (optional, default 4000) |
| `DATABASE_URL` | Supabase pooler connection string, `postgresql://…pooler.supabase.com:6543/postgres` |
| `SUPABASE_URL` | `https://<project-ref>.supabase.co` |
| `SUPABASE_SECRET_KEY` | Supabase secret (service-role) key; server only, never commit it |
| `CORS_ORIGIN` | optional; allowed origin(s) |

The server refuses to start if a variable is missing or malformed. To check that it's running:

```bash
curl http://localhost:4000/api/v1/health
# {"status":200,"success":true,"message":"OK","data":{"uptime":3,"database":"up"}}
```

Other scripts: `npm run typecheck`, `npm run build` / `npm start`, `npm run db:generate -- --name <change>`,
`npm run db:migrate`.

## ML quick start

Requires [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically.

```bash
cd ml
uv sync
uv run pytest
```

See [`ml/README.md`](ml/README.md) for the layout, packages and method rules.

## Docs

- [`docs/api.md`](docs/api.md): API contract consumed by the app
- [`docs/data-model.md`](docs/data-model.md): database tables
- [`docs/game-rules.md`](docs/game-rules.md): XP, streaks, hearts, leaderboard, offline sync

## Ground rules

- **Privacy:** user camera frames, video or hand landmarks are never accepted, stored or logged. Recognition runs
  on the device.
- **Content:** sign descriptions stay `verified = false` until they're checked against the NDFN NSL dictionary.
  We never invent NSL sign descriptions, and we say "Deaf" / "hard of hearing", never "deaf and mute".
- Never commit `.env` files, Supabase keys, datasets, extracted landmarks, model artefacts or videos.
- A contract change (endpoint, payload or game rule) needs a matching `ishara-app` change.
