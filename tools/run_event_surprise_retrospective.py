"""Run the frozen event-surprise retrospective from cache-only artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from sentiment_lab.data.storage import ArtifactStore, file_sha256
from sentiment_lab.event_surprise.retrospective import (
    RetrospectiveSpecification,
    run_retrospective,
)
from sentiment_lab.redesign.experiment import (
    RedesignConfig,
    configuration_hash,
    write_cache_only_run,
)


def _number(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _percent(value: Any, digits: int = 2) -> str:
    return "n/a" if value is None else f"{100 * float(value):.{digits}f}%"


def _display_path(path: Path, repository_root: Path) -> Path:
    try:
        return path.relative_to(repository_root)
    except ValueError:
        return path


def _render_report(
    metrics: dict[str, Any],
    *,
    config_digest: str,
    data_hashes: dict[str, str],
    run_directory: Path,
) -> str:
    primary = metrics["splits"]["holdout"]
    status = str(metrics["status"]).replace("_", " ").upper()
    gate_lines = [
        f"| {name.replace('_', ' ')} | {'PASS' if passed else 'FAIL'} |"
        for name, passed in metrics["promotion_gates"].items()
    ]
    split_lines: list[str] = []
    for split in ("development", "validation", "holdout"):
        values = metrics["splits"][split]
        interval = values["base_net_sharpe_block_bootstrap_95pct"]
        split_lines.append(
            "| "
            + " | ".join(
                [
                    split,
                    str(values["candidate_events"]),
                    str(values["evaluation_events"]),
                    str(values["accepted_trades"]),
                    _number(values["event_signal_spearman_5d"]),
                    _number(values["gross"]["sharpe"]),
                    _number(values["base_net"]["sharpe"]),
                    _number(values["conservative_net"]["sharpe"]),
                    f"[{_number(interval['lower'])}, {_number(interval['upper'])}]",
                    _percent(values["base_net"]["total_return"]),
                ]
            )
            + " |"
        )
    rejection_lines = [
        f"| {row['reason']} | {row['len']} |" for row in metrics["selection_counts"]["rejections"]
    ]
    return f"""# Event-Surprise Retrospective

## Decision

**{status}.** The frozen event-surprise portfolio {"passed every" if metrics["all_promotion_gates_passed"] else "did not pass all"} predeclared promotion gates. The primary holdout base-cost net Sharpe was **{_number(primary["base_net"]["sharpe"])}**; the conservative-cost net Sharpe was **{_number(primary["conservative_net"]["sharpe"])}**; and the five-session block-bootstrap 95% interval for base-cost net Sharpe was **[{_number(primary["base_net_sharpe_block_bootstrap_95pct"]["lower"])}, {_number(primary["base_net_sharpe_block_bootstrap_95pct"]["upper"])}]**.

This is a strategy decision, not a claim that NLP cannot classify financial text or help analysts. It asks only whether this specific event-surprise construction produced a sufficiently robust, costed daily portfolio in the frozen 5,000-article sample.

## Research question

Can a sparse signal based on the disagreement between a structured event-surprise assessment and FinBERT, scaled by company specificity, materiality, novelty, and confidence, support a tradable five-session long/short event portfolio after explicit costs?

The economic specification was frozen in `config/experiments/event_surprise_retrospective.yaml` before the first portfolio calculation. A pre-canonical verification run then exposed a boundary-handling defect: it retained the earlier split's boundary day and discarded valid terminal-holdout exits. The runner was corrected to purge outcomes reaching the next split and retain complete terminal-holdout paths. No horizon, threshold, weight, cost assumption, promotion gate, or model was changed in response to performance.

## Frozen portfolio specification

- 5,000 full-text articles, 125 companies, 2022-2026, with a chronological 60/20/20 development/validation/holdout split.
- Abstentions excluded; one event per company-day selected by greatest absolute signal, then article ID.
- Direction is the sign of `event_surprise_signal`; no post-freeze signal threshold search.
- Expected edge is scaled by a no-intercept OLS coefficient fitted only on development observations.
- Entry is the frozen next adjusted market open; exit is the adjusted close after five sessions including entry.
- Same-ticker overlaps and split-crossing labels are rejected.
- Each event sleeve is capped at 2% of $1 million; each side is capped at 50%; gross exposure is capped at 100%; volume participation is capped at 1%.
- A trade must have development-fitted expected gross edge at least 2x its estimated base round-trip cost.
- Base costs: $0.005/share with $1 minimum, 5 bps half-spread per side, 2 bps slippage per side, nonlinear volume impact, 3% annualized short borrow, and $0.02 research allocation per entry.
- Conservative costs: 10 bps half-spread, 5 bps slippage, doubled impact coefficient, and 8% borrow; commission and research allocation unchanged.
- Inactive market sessions remain in the daily return series as zero-return cash days.

## Results

| Split | Candidate events | Evaluation events | Accepted trades | Event IC (5d) | Gross Sharpe | Base net Sharpe | Conservative net Sharpe | Base Sharpe bootstrap 95% CI | Base net return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(split_lines)}

The development-only edge slope was `{metrics["development_edge_slope"]:.10f}` return units per signal unit. Across all splits, {metrics["selection_counts"]["accepted_trades"]} trades passed the cost and capacity rules.

## Promotion gates

| Gate | Result |
|---|---|
{chr(10).join(gate_lines)}

The gate thresholds were a minimum of 30 holdout trades, holdout base net Sharpe of at least 0.75, a block-bootstrap lower bound above zero, conservative holdout Sharpe above zero, positive validation base net return, and no single ticker contributing more than 20% of absolute holdout P&L.

## Reconciliation and rejection audit

| Rejection reason | Count |
|---|---:|
{chr(10).join(rejection_lines)}

Holdout base costs totaled `${primary["base_cost_usd"]:.2f}` and conservative costs totaled `${primary["conservative_cost_usd"]:.2f}`. Canonical output includes every prediction, requested order, fill, rejection, position-day, daily portfolio return, and decomposed cost row, allowing dollar P&L to be reconciled from source event through portfolio.

## Relation to the earlier broad-sentiment result

The earlier generic-sentiment five-session portfolio had gross Sharpe 1.1396, but base-cost net Sharpe 0.0082 and conservative-cost net Sharpe -1.6210. That result rejected the easy claim that positive language alone was a durable trading edge. This retrospective tested the narrower surprise-relative-to-expectations hypothesis without rewriting the prior result.

## Important limitations

- The event-level holdout IC was viewed before this portfolio specification was frozen. This is therefore a transparent one-shot final retrospective, not a pristine confirmatory holdout.
- The sample is intentionally balanced by company and is not a production universe or a capacity study.
- End-of-day prices cannot model intraday latency, queue position, or realized spreads.
- Borrow is modeled, but historical locate availability is unavailable; the test assumes every sampled short was locatable.
- The development-fitted linear edge scale is deliberately simple. No alternative calibrator was searched after the freeze.
- Qwen and FinBERT outputs are cached model assessments, not ground truth. Source-hash validation establishes lineage, not semantic correctness.
- Passing the gates would justify further prospective testing, not an alpha or deployability claim. Failing them closes this specification rather than inviting holdout tuning.

## Reproduce

```powershell
uv sync --locked --extra dev
uv run --locked python tools/run_event_surprise_retrospective.py
uv run --locked ruff check .
uv run --locked mypy src/sentiment_lab
uv run --locked pytest
```

The runner is cache-only: it does not call OpenAI, Ollama, EODHD, CUDA, or any network service. It verifies all three frozen input hashes and refuses to overwrite an existing canonical run.

## Evidence identity

- Configuration SHA-256: `{config_digest}`
- Signals SHA-256: `{data_hashes["signals"]}`
- Articles SHA-256: `{data_hashes["articles"]}`
- Prices SHA-256: `{data_hashes["prices"]}`
- Canonical local run: `{run_directory.as_posix()}`
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/experiments/event_surprise_retrospective.yaml"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/results/event_surprise_retrospective")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/EVENT_SURPRISE_RETROSPECTIVE_REPORT.md")
    )
    parser.add_argument(
        "--evidence-json",
        type=Path,
        default=Path("docs/evidence/event_surprise_retrospective_summary.json"),
    )
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else repository_root / args.config
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = RedesignConfig.model_validate(raw)
    spec = RetrospectiveSpecification.from_config(config)

    input_paths = {
        name: repository_root / str(config.dataset[f"{name}_path"])
        for name in ("signals", "articles", "prices")
    }
    data_hashes = {name: file_sha256(path) for name, path in input_paths.items()}
    expected_hashes = {
        name: str(config.dataset[f"{name}_sha256"]).lower()
        for name in ("signals", "articles", "prices")
    }
    if data_hashes != expected_hashes:
        raise RuntimeError(
            "Frozen input hash mismatch: "
            + json.dumps({"expected": expected_hashes, "actual": data_hashes}, sort_keys=True)
        )

    result = run_retrospective(
        pl.read_parquet(input_paths["signals"]),
        pl.read_parquet(input_paths["articles"]),
        pl.read_parquet(input_paths["prices"]),
        spec,
    )
    output_root = (
        args.output_root if args.output_root.is_absolute() else repository_root / args.output_root
    )
    run_paths = write_cache_only_run(
        config,
        root=output_root,
        repository_root=repository_root,
        data_hashes=data_hashes,
        feature_hashes={"event_surprise_signal": data_hashes["signals"]},
        model_metadata={
            "qwen": "qwen3.6:35b-a3b Q4_K_M (cached)",
            "finbert": "ProsusAI/finbert@4556d13015211d73dccd3fdd39d39232506f3e43 (cached)",
            "edge_calibration": "development-only OLS through origin",
        },
        frames=result.frames,
        metrics=result.metrics,
    )
    config_digest = configuration_hash(config)
    evidence = {
        "configuration_hash": config_digest,
        "data_hashes": data_hashes,
        "canonical_run": _display_path(run_paths.root, repository_root).as_posix(),
        "metrics": result.metrics,
    }
    evidence_path = (
        args.evidence_json
        if args.evidence_json.is_absolute()
        else repository_root / args.evidence_json
    )
    ArtifactStore(repository_root, repository_root / "data" / "research.duckdb").write_json(
        evidence, evidence_path
    )
    report_path = args.report if args.report.is_absolute() else repository_root / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        _render_report(
            result.metrics,
            config_digest=config_digest,
            data_hashes=data_hashes,
            run_directory=_display_path(run_paths.root, repository_root),
        ),
        encoding="utf-8",
    )
    print(json.dumps({"run": str(run_paths.root), "status": result.metrics["status"]}, indent=2))


if __name__ == "__main__":
    main()
