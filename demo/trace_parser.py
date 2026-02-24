import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from demo.schemas import RECORD_REQUIRED_KEYS, RECORD_TYPE_ORDER


@dataclass
class ParsedTrace:
    path: str
    records: List[Dict[str, Any]]
    by_type: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    @property
    def query_info(self) -> Dict[str, Any]:
        return (self.by_type.get("query_info") or [{}])[0]

    @property
    def final_answer(self) -> Dict[str, Any]:
        return (self.by_type.get("final_answer") or [{}])[0]

    @property
    def evaluation_metrics(self) -> Dict[str, Any]:
        return (self.by_type.get("evaluation_metrics") or [{}])[0]

    @property
    def performance_metrics(self) -> Dict[str, Any]:
        return (self.by_type.get("performance_metrics") or [{}])[0]

    @property
    def execution_results(self) -> List[Dict[str, Any]]:
        return self.by_type.get("execution_result", [])

    @property
    def fused_steps(self) -> List[Dict[str, Any]]:
        return self.by_type.get("fused_answer_step", [])

    def to_export_json(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "warnings": self.warnings,
            "records": self.records,
            "grouped": {k: self.by_type.get(k, []) for k in RECORD_TYPE_ORDER if self.by_type.get(k)},
        }


def _validate_record(record: Dict[str, Any], line_no: int) -> List[str]:
    warnings: List[str] = []
    rtype = record.get("type")
    if not isinstance(rtype, str):
        return [f"Line {line_no}: missing or invalid 'type' field"]
    expected = RECORD_REQUIRED_KEYS.get(rtype)
    if expected is None:
        warnings.append(f"Line {line_no}: unknown record type '{rtype}'")
        return warnings
    missing = sorted(k for k in expected if k not in record)
    if missing:
        warnings.append(f"Line {line_no}: record type '{rtype}' missing keys: {missing}")
    return warnings


def parse_trace_file(path: str) -> ParsedTrace:
    p = Path(path)
    records: List[Dict[str, Any]] = []
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    warnings: List[str] = []

    if not p.exists():
        return ParsedTrace(path=str(p), records=[], by_type={}, warnings=[f"Trace file not found: {p}"])

    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                warnings.append(f"Line {i}: invalid JSON ({e})")
                continue
            if not isinstance(record, dict):
                warnings.append(f"Line {i}: JSON record is not an object")
                continue
            warnings.extend(_validate_record(record, i))
            records.append(record)
            by_type.setdefault(record.get("type", "unknown"), []).append(record)

    return ParsedTrace(path=str(p), records=records, by_type=by_type, warnings=warnings)


def list_trace_files(output_dir: str) -> List[str]:
    p = Path(output_dir)
    if not p.exists():
        return []

    def _sort_key(fp: Path) -> int:
        stem = fp.stem  # query_1_results
        parts = stem.split("_")
        try:
            return int(parts[1])
        except (IndexError, ValueError):
            return 10**9

    files = sorted(p.glob("query_*_results.jsonl"), key=_sort_key)
    return [str(x) for x in files]


def summarize_routing_distribution(trace: ParsedTrace) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for rec in trace.execution_results:
        route = str(rec.get("routing", "unknown"))
        counts[route] = counts.get(route, 0) + 1
    return counts


def summarize_doc_source_distribution(trace: ParsedTrace) -> Dict[str, int]:
    """
    Best-effort only. Current JSONL docs usually contain {text, score} without source_id.
    Returns an empty dict in that case.
    """
    counts: Dict[str, int] = {}
    for rec in trace.execution_results:
        for doc in rec.get("docs", []) or []:
            if not isinstance(doc, dict):
                continue
            src = doc.get("source_id")
            if not src:
                continue
            counts[str(src)] = counts.get(str(src), 0) + 1
    return counts

