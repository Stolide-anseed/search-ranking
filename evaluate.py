from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from reranking.zero_shot import evaluate_zero_shot, load_config as load_reranker_config
from retrieval.query_knn_search import (
    load_query_knn_search_config,
    run_query_knn_search,
)


ROOT = Path(__file__).resolve().parent


def run_query_knn():
    config = load_query_knn_search_config(ROOT / "configs" / "query_knn_search.toml")
    result = run_query_knn_search(config)
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    return config, result


def run_reranker(oof_candidates: Path | None = None) -> dict[str, object]:
    config = load_reranker_config(ROOT / "configs" / "reranker.toml")
    if oof_candidates is not None:
        config = replace(
            config,
            oof_candidates=oof_candidates,
        )
    result = evaluate_zero_shot(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce grouped nested OOF metrics")
    parser.add_argument("command", choices=("query-knn", "reranker", "all"))
    args = parser.parse_args()
    if args.command == "query-knn":
        run_query_knn()
    elif args.command == "reranker":
        run_reranker()
    else:
        query_config, _ = run_query_knn()
        run_reranker(query_config.output_dir / "reranker_candidates_oof.f")


if __name__ == "__main__":
    main()
