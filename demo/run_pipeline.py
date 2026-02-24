import csv
import io
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rag.initializer import initialize_multi_source_rag, initialize_rag_system  # noqa: E402
from runner.main_rag_only import (  # noqa: E402
    get_save_dir,
    save_overall_results,
    single_query_execution,
)
from utils.data_load import (  # noqa: E402
    load_corpus_and_profiles,
    load_multi_source_corpus,
    load_multi_source_queries,
    load_queries,
)
from utils.metrics import calculate_overall_metrics  # noqa: E402


LAST_RUN_META_PATH = REPO_ROOT / "outputs" / ".demo_last_run.json"


@dataclass(frozen=True)
class PresetSpec:
    label: str
    dataset: str
    source_count: int
    is_multi_source_dataset: bool


PRESETS: Dict[str, PresetSpec] = {
    "2 source preset": PresetSpec(
        label="2 source preset", dataset="hotpot_qa", source_count=2, is_multi_source_dataset=False
    ),
    "3 source preset": PresetSpec(
        label="3 source preset", dataset="multi_source", source_count=3, is_multi_source_dataset=True
    ),
    "4 source preset": PresetSpec(
        label="4 source preset", dataset="mixed_4source", source_count=4, is_multi_source_dataset=True
    ),
}


@dataclass
class QueryItem:
    query: str
    ground_truth: str
    has_ground_truth: bool
    source: Optional[str] = None
    input_index: Optional[int] = None


@dataclass
class CustomSourceSpec:
    source_name: str
    source_profile: str
    docs: List[str]
    input_doc_count: int
    file_name: str = ""


@dataclass
class DemoConfig:
    preset_label: str
    mode: str  # "hard_routing" | "adaptive_cap"
    sample_size: int = 1
    query_index: Optional[int] = None  # 0-based index into prepared query list before sample_size crop
    rag_type: str = "naive"
    openai_model: str = "gpt-4o"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    decompose: bool = True
    use_reflection: bool = True
    max_reflexion_times: int = 2
    top_k_per_source: int = 8
    keep_k: int = 8
    selector: str = "score"
    per_source_cap: int = 0
    preferred_cap: int = 5
    other_cap: int = 2
    compare_with_baseline: bool = False
    custom_queries: Optional[List[QueryItem]] = None
    custom_source: Optional[CustomSourceSpec] = None


@dataclass
class SingleRunResult:
    label: str
    mode: str
    output_dir: str
    overall_metrics: Dict[str, Any]
    actual_run_count: int
    query_items: List[Dict[str, Any]]
    custom_mode: bool
    preset_label: str
    dataset: str
    params: Dict[str, Any]


@dataclass
class DemoRunResult:
    primary: SingleRunResult
    baseline: Optional[SingleRunResult] = None


_RESOURCE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


@contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_api_config(config: DemoConfig) -> Tuple[str, str, Optional[str]]:
    api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required.")
    model = config.openai_model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    base_url = config.openai_base_url or os.environ.get("OPENAI_API_BASE")
    return api_key, model, base_url


def parse_uploaded_queries(filename: str, raw_bytes: bytes) -> List[QueryItem]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return _parse_json_queries(raw_bytes)
    if suffix == ".csv":
        return _parse_csv_queries(raw_bytes)
    raise ValueError("Unsupported file type. Please upload JSON or CSV.")


def parse_uploaded_source_corpus(filename: str, raw_bytes: bytes) -> List[str]:
    """
    Parse a custom source corpus file.
    Supported formats:
      - JSON: [{"title": "...", "text": "..."}, ...] or ["doc text", ...]
      - CSV:  must contain `text` column, optional `title`
    Returns a list of document strings in the same format used by existing loaders:
      "title. text" (or just text if no title is available)
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".json":
        return _parse_source_json(raw_bytes)
    if suffix == ".csv":
        return _parse_source_csv(raw_bytes)
    raise ValueError("Unsupported custom source file type. Please upload JSON or CSV.")


def _parse_json_queries(raw_bytes: bytes) -> List[QueryItem]:
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        data = json.loads(raw_bytes.decode("utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("JSON must be a list of objects.")
    items: List[QueryItem] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"JSON row {idx} is not an object.")
        query = str(row.get("query", "")).strip()
        if not query:
            continue
        gt_raw = row.get("ground_truth")
        has_gt = gt_raw is not None and str(gt_raw).strip() != ""
        items.append(
            QueryItem(
                query=query,
                ground_truth=str(gt_raw).strip() if has_gt else "",
                has_ground_truth=has_gt,
                input_index=idx,
            )
        )
    if not items:
        raise ValueError("No valid query rows found in JSON.")
    return items


def _join_title_text(title: str, text: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    if not text:
        return ""
    if not title:
        return text
    if text.startswith(title):
        return text
    return f"{title}. {text}"


def _parse_source_json(raw_bytes: bytes) -> List[str]:
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except UnicodeDecodeError:
        data = json.loads(raw_bytes.decode("utf-8-sig"))

    if not isinstance(data, list):
        raise ValueError("Custom source JSON must be a list.")

    docs: List[str] = []
    for idx, row in enumerate(data):
        if isinstance(row, str):
            text = row.strip()
            if text:
                docs.append(text)
            continue
        if not isinstance(row, dict):
            raise ValueError(f"Custom source JSON row {idx} must be an object or string.")
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        title = str(row.get("title", "")).strip()
        doc = _join_title_text(title, text)
        if doc:
            docs.append(doc)

    if not docs:
        raise ValueError("No valid documents found in custom source JSON.")
    return docs


def _parse_csv_queries(raw_bytes: bytes) -> List[QueryItem]:
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("CSV must include a header row.")
    header_map = {h.strip().lower(): h for h in reader.fieldnames if h}
    if "query" not in header_map:
        raise ValueError("CSV must contain a 'query' column.")
    gt_key = header_map.get("ground_truth")
    q_key = header_map["query"]
    items: List[QueryItem] = []
    for idx, row in enumerate(reader):
        query = str(row.get(q_key, "")).strip()
        if not query:
            continue
        gt_raw = row.get(gt_key) if gt_key else None
        has_gt = gt_raw is not None and str(gt_raw).strip() != ""
        items.append(
            QueryItem(
                query=query,
                ground_truth=str(gt_raw).strip() if has_gt else "",
                has_ground_truth=has_gt,
                input_index=idx,
            )
        )
    if not items:
        raise ValueError("No valid query rows found in CSV.")
    return items


def _parse_source_csv(raw_bytes: bytes) -> List[str]:
    text = raw_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ValueError("Custom source CSV must include a header row.")

    header_map = {h.strip().lower(): h for h in reader.fieldnames if h}
    if "text" not in header_map:
        raise ValueError("Custom source CSV must contain a 'text' column.")
    text_key = header_map["text"]
    title_key = header_map.get("title")

    docs: List[str] = []
    for row in reader:
        text_val = str(row.get(text_key, "")).strip()
        if not text_val:
            continue
        title_val = str(row.get(title_key, "")).strip() if title_key else ""
        doc = _join_title_text(title_val, text_val)
        if doc:
            docs.append(doc)

    if not docs:
        raise ValueError("No valid documents found in custom source CSV.")
    return docs


def _get_or_init_resources(spec: PresetSpec, rag_type: str) -> Dict[str, Any]:
    key = (spec.dataset, rag_type)
    if key in _RESOURCE_CACHE:
        return _RESOURCE_CACHE[key]

    with _pushd(REPO_ROOT):
        if spec.is_multi_source_dataset:
            sources_docs, source_profiles = load_multi_source_corpus(spec.dataset)
            rag_sources = initialize_multi_source_rag(rag_type, sources_docs)
            resources = {
                "rag_sources": rag_sources,
                "source_profiles": source_profiles,
                "local_rag": None,
                "global_rag": None,
                "merged_rag": None,
                "local_profile": "",
                "global_profile": "",
            }
        else:
            local_docs, global_docs, local_profile, global_profile = load_corpus_and_profiles(spec.dataset)
            # Initialize routing-ready RAGs once. This supports both hard routing and multi-source (2-source) mode.
            local_rag, global_rag, _merged_unused = initialize_rag_system(rag_type, True, local_docs, global_docs)
            resources = {
                "rag_sources": None,
                "source_profiles": None,
                "local_rag": local_rag,
                "global_rag": global_rag,
                "merged_rag": None,
                "local_profile": local_profile,
                "global_profile": global_profile,
            }
    _RESOURCE_CACHE[key] = resources
    return resources


def _load_preset_queries(spec: PresetSpec, sample_size: Optional[int] = None) -> List[QueryItem]:
    with _pushd(REPO_ROOT):
        if spec.is_multi_source_dataset:
            rows = load_multi_source_queries(spec.dataset, sample_size)
        else:
            rows = load_queries(spec.dataset, sample_size)

    items: List[QueryItem] = []
    for idx, row in enumerate(rows):
        gt = str(row.get("ground_truth", ""))
        items.append(
            QueryItem(
                query=str(row["query"]),
                ground_truth=gt,
                has_ground_truth=gt.strip() != "",
                source=row.get("source"),
                input_index=idx,
            )
        )
    return items


def _prepare_queries(config: DemoConfig, spec: PresetSpec) -> List[QueryItem]:
    if config.custom_queries is not None:
        base = list(config.custom_queries)
        if config.query_index is not None:
            if config.query_index < 0 or config.query_index >= len(base):
                raise ValueError(f"query_index out of range: {config.query_index} (size={len(base)})")
            return [base[config.query_index]]
        return base[: max(1, config.sample_size)]

    # For preset mode, keep the repo's deterministic sampling behavior.
    if config.query_index is not None:
        full = _load_preset_queries(spec, sample_size=None)
        if config.query_index < 0 or config.query_index >= len(full):
            raise ValueError(f"query_index out of range: {config.query_index} (size={len(full)})")
        return [full[config.query_index]]
    return _load_preset_queries(spec, sample_size=max(1, config.sample_size))


def _query_dicts_for_runner(query_items: List[QueryItem]) -> List[Dict[str, Any]]:
    rows = []
    for q in query_items:
        row = {"query": q.query, "ground_truth": q.ground_truth}
        if q.source:
            row["source"] = q.source
        rows.append(row)
    return rows


def _mode_flags(spec: PresetSpec, mode: str, has_custom_source: bool = False) -> Dict[str, Any]:
    if mode not in {"hard_routing", "adaptive_cap"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if mode == "hard_routing":
        if spec.is_multi_source_dataset or has_custom_source:
            return {
                "multi_source": True,
                "hard_routing_multi": True,
                "use_routing": False,
                "preferred_cap": 0,
                "other_cap": 0,
                "per_source_cap": 0,
            }
        return {
            "multi_source": False,
            "hard_routing_multi": False,
            "use_routing": True,
            "preferred_cap": 0,
            "other_cap": 0,
            "per_source_cap": 0,
        }

    # adaptive_cap
    return {
        "multi_source": True,
        "hard_routing_multi": False,
        "use_routing": False,
        "preferred_cap": 0,
        "other_cap": 0,
        "per_source_cap": 0,
    }


def _build_output_dir(config: DemoConfig, spec: PresetSpec, mode: str, timestamp_tag: str) -> str:
    flags = _mode_flags(spec, mode, has_custom_source=(config.custom_source is not None))
    multi_source = flags["multi_source"]
    hard_routing_multi = flags["hard_routing_multi"]
    use_routing = flags["use_routing"]

    if mode == "adaptive_cap":
        preferred_cap = config.preferred_cap
        other_cap = config.other_cap
        per_source_cap = 0
    else:
        preferred_cap = 0
        other_cap = 0
        per_source_cap = 0

    base = get_save_dir(
        config.decompose,
        use_routing,
        config.use_reflection,
        spec.dataset,
        config.rag_type,
        multi_source=multi_source,
        hard_routing_multi=hard_routing_multi,
        keep_k=config.keep_k,
        per_source_cap=per_source_cap,
        model=config.openai_model,
        selector=config.selector,
        preferred_cap=preferred_cap,
        other_cap=other_cap,
    )

    suffix = "__demo_custom" if config.custom_queries is not None else "__demo_preset"
    if config.custom_source is not None:
        suffix += "_with_custom_source"
    return f"{base}{suffix}_{mode}_{timestamp_tag}"


def _merge_custom_source_into_resources(
    spec: PresetSpec,
    base_resources: Dict[str, Any],
    config: DemoConfig,
) -> Dict[str, Any]:
    if config.custom_source is None:
        return base_resources

    custom = config.custom_source
    source_name = (custom.source_name or "").strip()
    source_profile = (custom.source_profile or "").strip()
    if not source_name:
        raise ValueError("Custom source name is required when custom source upload is enabled.")
    if not source_profile:
        raise ValueError("Custom source profile is required when custom source upload is enabled.")
    if not custom.docs:
        raise ValueError("Custom source has no documents.")

    with _pushd(REPO_ROOT):
        custom_rag = initialize_multi_source_rag(config.rag_type, {source_name: custom.docs})[source_name]

    if spec.is_multi_source_dataset:
        rag_sources = dict(base_resources["rag_sources"] or {})
        source_profiles = dict(base_resources["source_profiles"] or {})
        if source_name in rag_sources or source_name in source_profiles:
            raise ValueError(f"Custom source name '{source_name}' conflicts with an existing source. Please choose another name.")
        rag_sources[source_name] = custom_rag
        source_profiles[source_name] = source_profile
        return {
            **base_resources,
            "rag_sources": rag_sources,
            "source_profiles": source_profiles,
        }

    # 2-source preset path: synthesize an N-source view for multi-source retrieval / N-way hard routing.
    rag_sources = {
        "local": base_resources["local_rag"],
        "global": base_resources["global_rag"],
        source_name: custom_rag,
    }
    source_profiles = {
        "local": base_resources["local_profile"],
        "global": base_resources["global_profile"],
        source_name: source_profile,
    }
    return {
        **base_resources,
        "rag_sources": rag_sources,
        "source_profiles": source_profiles,
    }


def _run_once(config: DemoConfig, spec: PresetSpec, mode: str, timestamp_tag: str) -> SingleRunResult:
    api_key, model, base_url = _resolve_api_config(config)
    query_items = _prepare_queries(config, spec)
    queries_for_runner = _query_dicts_for_runner(query_items)
    flags = _mode_flags(spec, mode, has_custom_source=(config.custom_source is not None))

    # Fill adaptive-cap-specific flags
    if mode == "adaptive_cap":
        flags["preferred_cap"] = max(0, int(config.preferred_cap))
        flags["other_cap"] = max(0, int(config.other_cap))
        flags["per_source_cap"] = 0

    output_dir_rel = _build_output_dir(config, spec, mode, timestamp_tag)
    output_dir_abs = str((REPO_ROOT / output_dir_rel).resolve())

    base_resources = _get_or_init_resources(spec, config.rag_type)
    resources = _merge_custom_source_into_resources(spec, base_resources, config)

    with _pushd(REPO_ROOT):
        os.makedirs(output_dir_rel, exist_ok=True)
        all_metrics: List[Dict[str, Any]] = []
        all_metrics = single_query_execution(
            config.decompose,
            all_metrics,
            queries_for_runner,
            resources["local_rag"],
            resources["global_rag"],
            resources["merged_rag"],
            flags["use_routing"],
            config.use_reflection,
            config.max_reflexion_times,
            resources["local_profile"],
            resources["global_profile"],
            api_key,
            model,
            base_url,
            output_dir_rel,
            flags["multi_source"],
            config.top_k_per_source,
            config.keep_k,
            config.selector,
            per_source_cap=flags.get("per_source_cap", 0),
            rag_sources=resources["rag_sources"],
            source_profiles=resources["source_profiles"],
            hard_routing_multi=flags["hard_routing_multi"],
            preferred_cap=flags.get("preferred_cap", 0),
            other_cap=flags.get("other_cap", 0),
        )
        overall_metrics = calculate_overall_metrics(all_metrics)
        save_overall_results(output_dir_rel, overall_metrics, queries_for_runner, all_metrics)

        sidecar = {
            "preset_label": spec.label,
            "dataset": spec.dataset,
            "mode": mode,
            "custom_mode": config.custom_queries is not None,
            "actual_run_count": len(query_items),
            "query_items": [asdict(q) for q in query_items],
            "params": {
                "decompose": config.decompose,
                "use_reflection": config.use_reflection,
                "max_reflexion_times": config.max_reflexion_times,
                "top_k_per_source": config.top_k_per_source,
                "keep_k": config.keep_k,
                "selector": config.selector,
                "preferred_cap": config.preferred_cap if mode == "adaptive_cap" else 0,
                "other_cap": config.other_cap if mode == "adaptive_cap" else 0,
                "rag_type": config.rag_type,
                "openai_model": model,
                "custom_source": {
                    "enabled": config.custom_source is not None,
                    "source_name": config.custom_source.source_name if config.custom_source else None,
                    "doc_count": len(config.custom_source.docs) if config.custom_source else 0,
                },
            },
            "overall_metrics": overall_metrics,
        }
        with open(Path(output_dir_rel) / "demo_run_meta.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, ensure_ascii=False, indent=2)

    return SingleRunResult(
        label="ours" if mode == "adaptive_cap" else "baseline",
        mode=mode,
        output_dir=output_dir_abs,
        overall_metrics=overall_metrics,
        actual_run_count=len(query_items),
        query_items=[asdict(q) for q in query_items],
        custom_mode=config.custom_queries is not None,
        preset_label=spec.label,
        dataset=spec.dataset,
        params=sidecar["params"],
    )


def run_demo(config: DemoConfig) -> DemoRunResult:
    if config.preset_label not in PRESETS:
        raise ValueError(f"Unknown preset: {config.preset_label}")
    spec = PRESETS[config.preset_label]

    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    primary = _run_once(config, spec, config.mode, timestamp_tag)
    baseline = None

    if config.compare_with_baseline:
        other_mode = "hard_routing" if config.mode == "adaptive_cap" else "adaptive_cap"
        baseline = _run_once(config, spec, other_mode, f"{timestamp_tag}_cmp")
        # Normalize labels for UI semantics
        if baseline.mode == "hard_routing":
            baseline.label = "baseline"
        else:
            baseline.label = "ours"
        if primary.mode == "hard_routing":
            primary.label = "baseline"
        else:
            primary.label = "ours"

    result = DemoRunResult(primary=primary, baseline=baseline)
    _write_last_run_meta(result)
    return result


def _write_last_run_meta(result: DemoRunResult) -> None:
    LAST_RUN_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "primary": asdict(result.primary),
        "baseline": asdict(result.baseline) if result.baseline else None,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    with LAST_RUN_META_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_last_run_meta() -> Optional[Dict[str, Any]]:
    if not LAST_RUN_META_PATH.exists():
        return None
    try:
        with LAST_RUN_META_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_run_sidecar(output_dir: str) -> Optional[Dict[str, Any]]:
    p = Path(output_dir) / "demo_run_meta.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
