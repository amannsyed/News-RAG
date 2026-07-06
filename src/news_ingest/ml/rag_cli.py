from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("news_ingest.ml.rag_api:app", host="0.0.0.0", port=8003, reload=False)


if __name__ == "__main__":
    main()
