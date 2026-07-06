# News RAG UI

React/Vite chat UI for the local News RAG API.

## Run Locally

Start the backend first from the repository root:

```bash
docker compose up -d postgres embedding-worker rag-api
```

Then start the UI:

```bash
cd ui
npm install
npm run dev
```

Open the Vite URL shown in the terminal, normally `http://localhost:5173`. The Vite dev proxy forwards `/api/*` requests to `http://localhost:8003`.

## Auth Token

The UI does not bundle a hard-coded API token. For local development, either paste the `RAG_API_TOKEN` into the API token field in the sidebar, or create `ui/.env.local`:

```bash
VITE_RAG_API_TOKEN=your-local-rag-token
```

Values prefixed with `VITE_` are included in the browser bundle, so use this only for local development. Production should put auth behind a backend session or proxy.

## Checks

```bash
npm run lint
npm run build
```

## Chat Behavior

The UI sends the default `user_id` value `user_id`. Chat requests default to 20 RAG articles, and when Web Search is enabled the UI asks for 20 web results. Follow-up questions returned by the API are shown as clickable chips; clicking a chip only fills the input.
