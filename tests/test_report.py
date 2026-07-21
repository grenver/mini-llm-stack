"""Report-generation integrity: populated JSON must never render blank.

Regression guard for a real shipped bug: bench rows with a different field
set than the FIRST row (Phase 4 emitted memory rows then latency rows)
rendered as blank table cells, silently dropping every latency number while
the README quoted them. These tests fail loudly if any value in any results
file does not survive into report.md.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))

import make_report  # noqa: E402
from make_report import fmt_table, group_rows  # noqa: E402

RESULTS = Path(__file__).parent.parent / "bench" / "results"


def test_heterogeneous_rows_split_into_full_tables():
    rows = [
        {"kind": "memory", "variant": "fp32", "weight_mb": 97.0},
        {"kind": "latency", "M": 1, "impl": "fused_int8", "latency_ms": 1.2},
    ]
    out = fmt_table(rows)
    # every field name and every value of every row must be rendered
    for r in rows:
        for k, v in r.items():
            assert k in out, f"column {k} missing"
            assert str(v) in out, f"value {v} missing"
    assert len(group_rows(rows)) == 2


def test_no_blank_cells_for_present_fields():
    rows = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    out = fmt_table(rows)
    assert "| 1 | 2 |" in out and "| 3 | 4 |" in out


@pytest.mark.skipif(not any(RESULTS.glob("*.json")),
                    reason="no benchmark results present")
def test_every_result_value_survives_into_report():
    """End-to-end drift check across ALL result files: regenerate the report
    and require every row value from every JSON to appear in it."""
    make_report.main()
    report = (RESULTS.parent / "report.md").read_text(encoding="utf-8")
    problems = []
    for f in sorted(RESULTS.glob("*.json")):
        blob = json.loads(f.read_text())
        for row in blob.get("rows", []):
            for k, v in row.items():
                if str(v) not in report:
                    problems.append(f"{f.name}: row value {k}={v!r} "
                                    "not rendered")
    assert not problems, "\n".join(problems[:10])
