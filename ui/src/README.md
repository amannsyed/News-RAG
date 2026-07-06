# UI Source

React/Vite source for the News RAG web interface.

## Files

- `App.jsx`: main chat/search UI, model controls, thinking controls, web-search toggle, citations, follow-up questions, and conversation delete actions.
- `App.css`: application layout and component styling.
- `main.jsx`: React entry point.
- `index.css`: global browser styles.
- `assets/`: local images and SVG assets used by the UI.

## Local Commands

From the repository root:

```bash
npm run dev --prefix ui
npm run lint --prefix ui
npm run build --prefix ui
```

The UI talks to the RAG API through `VITE_RAG_API_URL`. For local-only development, you can paste the API token in the sidebar or set `VITE_RAG_API_TOKEN` in `ui/.env.local`.
