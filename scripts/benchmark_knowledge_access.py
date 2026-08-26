"""Small deterministic benchmark for EV-MEMORY-01 read-side primitives.

Usage from the repository root::

    python scripts/benchmark_knowledge_access.py --iterations 20

The output is measurement only; it never writes Truth or a Wiki file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time

from everos.memory.knowledge import (
    AuthorityState,
    TruthAwareQuery,
    TruthAwareRetriever,
    TruthClaim,
    TruthClass,
    TruthView,
    WikiCompiler,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    if args.iterations < 1:
        raise SystemExit("--iterations must be positive")

    benchmark_timestamp = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    claims = tuple(
        TruthClaim(
            claim_id=f"claim-{index:04d}",
            text=f"Project release fact {index}",
            kind="PROJECT_STATE",
            scope={"app_id": "bench", "project_id": "p", "owner_id": "u"},
            authority=AuthorityState.ACCEPTED,
            truth_class=TruthClass.CURRENT,
            source_refs=(f"episode-{index:04d}",),
            created_at=benchmark_timestamp,
            semantic_score=(index % 10) / 10,
            bm25_score=((index + 3) % 10) / 10,
        )
        for index in range(100)
    )
    request = TruthAwareQuery(
        query="release fact",
        scope={"app_id": "bench", "project_id": "p", "owner_id": "u"},
        view=TruthView.CURRENT,
        top_k=10,
    )
    retriever = TruthAwareRetriever()
    compiler = WikiCompiler()
    retrieval_times: list[float] = []
    wiki_times: list[float] = []
    rendered_count = 0
    for _ in range(args.iterations):
        started = time.perf_counter()
        hits = retriever.retrieve(claims, request)
        retrieval_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        page = compiler.compile(
            snapshot_id="benchmark-snapshot",
            title="Benchmark",
            claims=claims,
            scope=request.scope,
        )
        wiki_times.append((time.perf_counter() - started) * 1000)
        rendered_count = len(hits) + page.rendered_claim_count

    print(
        json.dumps(
            {
                "iterations": args.iterations,
                "claims": len(claims),
                "retrieval_ms_avg": sum(retrieval_times) / len(retrieval_times),
                "retrieval_ms_max": max(retrieval_times),
                "wiki_build_ms_avg": sum(wiki_times) / len(wiki_times),
                "wiki_build_ms_max": max(wiki_times),
                "rendered_count_check": rendered_count,
                "unsupported_claims": page.unsupported_claim_count,
                "stale_claims": page.stale_claim_count,
                "deterministic_hash": page.claim_set_sha256,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
