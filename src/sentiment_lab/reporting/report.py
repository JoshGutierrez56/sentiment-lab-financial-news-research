"""Compact, self-contained HTML report for the mandatory milestone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
from jinja2 import BaseLoader, Environment, select_autoescape

_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{{ title }}</title>
<style>
body{font:15px/1.45 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#17202a}
h1,h2{line-height:1.15} .meta{color:#58636f}.cards{display:flex;gap:12px;flex-wrap:wrap}.card{border:1px solid #d9dee3;border-radius:8px;padding:12px 16px;min-width:165px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e8eb;text-align:left;vertical-align:top;padding:8px}th{background:#f5f7f8;position:sticky;top:0}
.bullish{color:#08783e}.bearish{color:#a52626}.neutral{color:#59636e}.scroll{overflow:auto}.reason{min-width:240px}.article{max-width:420px;white-space:pre-wrap}.warn{background:#fff7df;border-left:4px solid #e5a000;padding:10px 14px}
</style></head><body>
<h1>{{ title }}</h1><p class="meta">Experiment {{ experiment_id }} · ticker {{ ticker }} · generated {{ generated_at }}</p>
<p class="warn">{{ metrics.definition }}</p>
{% for horizon, metric in metrics.horizons.items() %}
<h2>{{ horizon }} result</h2><div class="cards">
<div class="card"><strong>N</strong><br>{{ metric.n }}</div>
<div class="card"><strong>Directional accuracy</strong><br>{{ metric.directional_accuracy }}</div>
<div class="card"><strong>Spearman IC</strong><br>{{ metric.information_coefficient_spearman }}</div>
<div class="card"><strong>IC p-value</strong><br>{{ metric.information_coefficient_p_value }}</div>
<div class="card"><strong>Confidence-weighted IC</strong><br>{{ metric.confidence_weighted_ic_spearman }}</div>
</div>{% endfor %}
<h2>Article-level evidence</h2><div class="scroll"><table><thead><tr><th>Published (UTC)</th><th>Article</th><th>Assessment</th><th>Confidence</th><th>Reasoning</th>{% for h in horizons %}<th>{{ h }}d return</th>{% endfor %}</tr></thead><tbody>
{% for row in rows %}<tr><td>{{ row.publication_timestamp_utc }}</td><td><strong>{{ row.title }}</strong><div class="article">{{ row.article_text }}</div></td><td class="{{ row.sentiment_label }}">{{ row.sentiment_label }} ({{ row.sentiment_score }})</td><td>{{ row.confidence }}</td><td class="reason">{{ row.reasoning }}</td>{% for h in horizons %}<td>{{ row['future_return_' ~ h ~ 'd'] }}</td>{% endfor %}</tr>{% endfor %}
</tbody></table></div>
<h2>Reproducibility</h2><pre>{{ reproduction }}</pre>
</body></html>"""


def build_milestone_report(
    events: pl.DataFrame,
    metrics: dict[str, Any],
    *,
    output_path: str | Path,
    experiment_id: str,
    ticker: str,
    horizons: list[int],
    generated_at: str,
) -> Path:
    rows: list[dict[str, Any]] = []
    for row in events.to_dicts():
        rendered = dict(row)
        rendered["article_text"] = str(rendered.get("article_text", ""))[:800]
        for horizon in horizons:
            key = f"future_return_{horizon}d"
            value = rendered.get(key)
            rendered[key] = "—" if value is None else f"{float(value):.3%}"
        rendered["sentiment_score"] = f"{float(rendered['sentiment_score']):.2f}"
        rendered["confidence"] = f"{float(rendered['confidence']):.0%}"
        rows.append(rendered)
    environment = Environment(
        loader=BaseLoader(),
        autoescape=select_autoescape(default_for_string=True),
    )
    html = environment.from_string(_TEMPLATE).render(
        title="EODHD → ChatGPT → Future Returns Milestone",
        experiment_id=experiment_id,
        ticker=ticker,
        generated_at=generated_at,
        metrics=metrics,
        rows=rows,
        horizons=horizons,
        reproduction=json.dumps(
            {
                "command": "sentiment-lab milestone run --config config/experiments/milestone.yaml",
                "cached_rerun": "repeat the command without --refresh or --force-classify",
            },
            indent=2,
        ),
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path
