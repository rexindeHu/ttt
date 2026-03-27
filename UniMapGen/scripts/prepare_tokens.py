from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unimapgen.data.dataset import build_user_text, extract_assistant_lines, load_jsonl, sanitize_lines  # noqa: E402
from unimapgen.token_format import DiscreteMapTokenFormatter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare discrete token training targets without loading model weights.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--summary-json", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=896)
    parser.add_argument("--coord-num-bins", type=int, default=896)
    parser.add_argument("--token-schema", type=str, default="shared_numbers", choices=["legacy_xy", "shared_numbers"])
    parser.add_argument("--categories", type=str, default="auto")
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--token-sep", type=str, default="none", choices=["none", "space"])
    parser.add_argument("--include-system", action="store_true")
    parser.add_argument("--disable-legacy-text-prompt-tokens", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _first_user_text(sample: Dict[str, Any]) -> str:
    for msg in sample.get("messages", []):
        if str(msg.get("role", "")).strip().lower() == "user":
            return build_user_text(str(msg.get("content", "")))
    return ""


def _resolve_categories(rows: List[Dict[str, Any]], categories_arg: str) -> List[str]:
    raw = str(categories_arg).strip()
    if raw and raw.lower() != "auto":
        explicit = [item.strip() for item in raw.split(",") if item.strip()]
        if explicit:
            return explicit

    discovered: List[str] = []
    seen = set()
    for sample in rows:
        try:
            lines = sanitize_lines(extract_assistant_lines(sample))
        except Exception:
            continue
        for line in lines:
            category = str(line.get("category", "")).strip()
            if category and category not in seen:
                seen.add(category)
                discovered.append(category)
    return discovered or ["road"]


def _serialize_tokens(formatter: DiscreteMapTokenFormatter, lines: List[Dict[str, Any]], token_sep: str) -> str:
    spaced = formatter.lines_to_text(lines)
    known_tokens = formatter.extract_known_tokens(spaced)
    if token_sep == "space":
        return " ".join(known_tokens)
    return "".join(known_tokens)


def main() -> None:
    args = parse_args()

    input_jsonl = Path(args.input_jsonl).resolve()
    output_jsonl = Path(args.output_jsonl).resolve()
    summary_json = Path(args.summary_json).resolve()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(input_jsonl)
    if not rows:
        raise ValueError(f"No rows loaded from {input_jsonl}")

    resolved_categories = _resolve_categories(rows=rows, categories_arg=str(args.categories))

    formatter = DiscreteMapTokenFormatter(
        image_size=int(args.image_size),
        categories=resolved_categories,
        max_seq_len=int(args.max_seq_len),
        coord_num_bins=int(args.coord_num_bins),
        coordinate_token_style=str(args.token_schema),
        include_text_prompt_tokens=not bool(args.disable_legacy_text_prompt_tokens),
    )

    total = 0
    kept = 0
    dropped = 0
    dropped_ids: List[str] = []
    total_input_lines = 0
    total_kept_lines = 0

    with output_jsonl.open("w", encoding="utf-8") as out:
        for idx, sample in enumerate(rows):
            total += 1
            sample_id = str(sample.get("id", idx))

            try:
                raw_lines = extract_assistant_lines(sample)
            except Exception as exc:
                if args.strict:
                    raise
                dropped += 1
                dropped_ids.append(f"{sample_id}:missing_assistant:{exc}")
                continue

            lines = sanitize_lines(raw_lines)
            total_input_lines += len(raw_lines)
            total_kept_lines += len(lines)
            if not lines:
                if args.strict:
                    raise ValueError(f"Sample {sample_id} has no valid lines after sanitize.")
                dropped += 1
                dropped_ids.append(f"{sample_id}:empty_lines")
                continue

            token_text = _serialize_tokens(formatter=formatter, lines=lines, token_sep=str(args.token_sep))
            decoded_lines = sanitize_lines(formatter.text_to_lines(token_text))

            if not token_text.strip() or not decoded_lines:
                if args.strict:
                    raise ValueError(f"Sample {sample_id} token conversion failed.")
                dropped += 1
                dropped_ids.append(f"{sample_id}:tokenize_failed")
                continue

            user_text = _first_user_text(sample)
            rel_images = list(sample.get("images", []))
            messages: List[Dict[str, Any]] = []
            if args.include_system:
                messages.append({"role": "system", "content": formatter.build_system_prompt()})
            messages.append({"role": "user", "content": user_text})
            messages.append({"role": "assistant", "content": token_text})

            train_row: Dict[str, Any] = {
                "id": sample_id,
                "images": rel_images,
                "messages": messages,
            }
            out.write(json.dumps(train_row, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "total_samples": total,
        "kept_samples": kept,
        "dropped_samples": dropped,
        "total_input_lines": total_input_lines,
        "total_kept_lines": total_kept_lines,
        "token_schema": str(args.token_schema),
        "coord_num_bins": int(args.coord_num_bins),
        "image_size": int(args.image_size),
        "categories": resolved_categories,
        "categories_from": "dataset_auto" if str(args.categories).strip().lower() == "auto" else "args",
        "token_sep": str(args.token_sep),
        "include_system": bool(args.include_system),
        "dropped_ids": dropped_ids[:100],
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
