"""Evaluate a bootstrap prompt adapter before it can be selected."""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from peerpixel.lora_manifest import load_manifest, write_manifest


PHRASES = ("cinematic", "volumetric lighting", "watercolor", "masterpiece",
           "highly detailed", "8k", "award-winning")
COLORS = ("red", "orange", "yellow", "green", "blue", "purple", "pink",
          "black", "white", "gray", "grey", "brown")
NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
           "six": 6, "seven": 7, "eight": 8}


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _plain(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    return bool(stripped) and not stripped.startswith(("{", "[", "```")) and not any(
        lowered.startswith(prefix) for prefix in ("here is", "enhanced prompt", "prompt:"))


def _constraints(prompt: str) -> list[str]:
    lower = prompt.lower()
    values = re.findall(r'["“]([^"”]+)["”]', prompt)
    values += [color for color in COLORS if re.search(rf"\b{color}\b", lower)]
    values += [word for word in NUMBERS if re.search(rf"\b{word}\b", lower)]
    return values


def evaluate_outputs(cases: list[dict], outputs: list[str]) -> dict:
    if not cases or len(cases) != len(outputs):
        raise ValueError("cases and outputs must have the same nonzero length")
    count = len(cases)
    structure = sum(_plain(output) for output in outputs) / count
    fidelity_scores = []
    for case, output in zip(cases, outputs):
        required = _constraints(case["prompt"])
        lower = output.lower()
        fidelity_scores.append(not required or all(value.lower() in lower for value in required))
    fidelity = sum(fidelity_scores) / count
    within_limit = sum(len(output.split()) <= 192 for output in outputs) / count
    copies = sum(_normalized(case["prompt"]) == _normalized(output)
                 for case, output in zip(cases, outputs)) / count
    normalized_outputs = [_normalized(output) for output in outputs]
    distinct = len(set(normalized_outputs)) / count
    occurrences = Counter()
    for output in normalized_outputs:
        for phrase in PHRASES:
            if _normalized(phrase) in output:
                occurrences[phrase] += 1
    concentration = max(occurrences.values(), default=0) / count
    report = {
        "cases": count,
        "structureCompliance": structure,
        "promptFidelity": fidelity,
        "withinTokenLimit": within_limit,
        "normalizedCopyRate": copies,
        "distinctRate": distinct,
        "phraseConcentration": concentration,
    }
    report["passed"] = (structure >= 0.98 and fidelity == 1.0 and within_limit == 1.0
                        and copies <= 0.10 and distinct >= 0.95 and concentration <= 0.50)
    return report


def _read_cases(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if rows and "messages" in rows[0]:
        return [{"prompt": row["messages"][0]["content"],
                 "target": row["messages"][1]["content"]} for row in rows]
    return rows


def generate_cases(model_path: str, adapter: Path, cases: list[dict], batch_size: int = 8) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_path, local_files_only=True, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    outputs = []
    for offset in range(0, len(cases), batch_size):
        batch = cases[offset:offset + batch_size]
        chats = [tokenizer.apply_chat_template(
            [{"role": "user", "content": case["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for case in batch]
        encoded = tokenizer(chats, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            result = model.generate(**encoded, max_new_tokens=192, do_sample=False)
        start = encoded.input_ids.shape[1]
        outputs.extend(tokenizer.decode(row[start:], skip_special_tokens=True).strip()
                       for row in result)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--regression", required=True, type=Path)
    args = parser.parse_args()
    validation = _read_cases(args.validation)
    regression = _read_cases(args.regression)
    cases = validation + regression
    outputs = generate_cases(args.model, args.adapter, cases)
    report = evaluate_outputs(cases, outputs)
    report["validationCases"] = len(validation)
    report["regressionCases"] = len(regression)
    report["samples"] = [{**case, "output": output} for case, output in zip(cases, outputs)]
    (args.adapter / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    manifest = load_manifest(args.adapter / "manifest.json")
    manifest.pop("artifactFiles", None)
    manifest["evaluation"] = {key: value for key, value in report.items() if key != "samples"}
    write_manifest(args.adapter, manifest)
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
