"""Run the local event review service without starting audio capture."""

import argparse

import uvicorn

from dashboard.server import app, configure_review_store
from src.output.review import ReviewStore


def main():
    parser = argparse.ArgumentParser(description="Video-only local event review service")
    parser.add_argument("--db", default="reviews/candidates.sqlite3")
    parser.add_argument("--evidence-dir", default="evidence")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    configure_review_store(ReviewStore(args.db, args.evidence_dir))
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
