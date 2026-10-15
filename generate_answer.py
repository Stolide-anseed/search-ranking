from __future__ import annotations

import argparse
from pathlib import Path

from retrieval.io import read_dataframe
from submission import build_answer, load_bundled_scores, load_manifest


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ranked article results in answer.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "answer.csv")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="recompute BGE-M3 retrieval and GTE cross-encoder scores",
    )
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="rebuild the retrieval cache; valid only with --rebuild",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="reprocess raw article HTML before rebuilding; valid only with --rebuild",
    )
    parser.add_argument(
        "--force-model",
        action="store_true",
        help="ignore the validated GTE score cache; valid only with --rebuild",
    )
    args = parser.parse_args()
    if args.force_index and not args.rebuild:
        parser.error("--force-index requires --rebuild")
    if args.reprocess and not args.rebuild:
        parser.error("--reprocess requires --rebuild")
    if args.force_model and not args.rebuild:
        parser.error("--force-model requires --rebuild")

    manifest, inference_config = load_manifest(ROOT)
    if args.rebuild:
        try:
            from rebuild import rebuild_production_scores
        except ImportError as error:
            raise SystemExit(
                "Full dependencies are missing. Install requirements.txt first."
            ) from error
        scores = rebuild_production_scores(
            ROOT,
            force_index=args.force_index or args.reprocess,
            reprocess=args.reprocess,
            inference_config=inference_config,
            manifest=manifest,
            force_model=args.force_model,
        )
    else:
        scores = load_bundled_scores(ROOT, manifest)

    test_queries = read_dataframe(ROOT / "data" / "test.f")
    articles = read_dataframe(ROOT / "data" / "articles.f")
    answer = build_answer(scores, test_queries, articles, inference_config)
    output = args.output if args.output.is_absolute() else Path.cwd() / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    answer.to_csv(output, index=False)
    print(f"Generated {output.resolve()}: {len(answer)} queries, 10 articles per query")


if __name__ == "__main__":
    main()
