"""Train the disposable structure-only prompt adapter."""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from peerpixel.lora_manifest import safe_version, write_manifest


@dataclass(frozen=True)
class BootstrapTrainingConfig:
    model: str
    train: Path
    validation: Path
    output: Path
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    learning_rate: float = 5e-5
    epochs: float = 1.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_length: int = 512
    seed: int = 2468


class MessageDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        messages = self.rows[index]["messages"]
        prompt = self.tokenizer.apply_chat_template(
            messages[:1], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        complete = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False)
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = self.tokenizer(complete, add_special_tokens=False, truncation=True,
                                 max_length=self.max_length)
        labels = list(encoded["input_ids"])
        labels[:min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"],
                "labels": labels}


class CompletionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        width = max(len(row["input_ids"]) for row in rows)
        result = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = width - len(row["input_ids"])
            result["input_ids"].append(row["input_ids"] + [self.tokenizer.pad_token_id] * padding)
            result["attention_mask"].append(row["attention_mask"] + [0] * padding)
            result["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in result.items()}


def _dataset_manifest(path: Path) -> dict:
    manifest = json.loads((path.parent / "dataset-manifest.json").read_text(encoding="utf-8"))
    return {
        "trainDigest": manifest["digests"]["train"],
        "validationDigest": manifest["digests"]["validation"],
        "trainRecords": manifest["records"]["train"],
        "validationRecords": manifest["records"]["validation"],
    }


def train(config: BootstrapTrainingConfig) -> Path:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise RuntimeError("install PeerPixel's train optional dependencies first") from error
    version = safe_version(config.output.name)
    partial = config.output.with_name(version + ".partial")
    if config.output.exists() or partial.exists():
        raise FileExistsError("refusing to overwrite an adapter run")
    partial.mkdir(parents=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model, local_files_only=True,
                                                   padding_side="right")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            config.model, local_files_only=True, dtype=torch.bfloat16, device_map="cuda")
        model.config.use_cache = False
        model = get_peft_model(model, LoraConfig(
            r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ))
        training = MessageDataset(config.train, tokenizer, config.max_length)
        validation = MessageDataset(config.validation, tokenizer, config.max_length)
        arguments = TrainingArguments(
            output_dir=str(partial / "checkpoints"), per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.epochs, learning_rate=config.learning_rate,
            bf16=True, tf32=True, gradient_checkpointing=True,
            eval_strategy="steps", eval_steps=20, save_strategy="steps", save_steps=20,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="eval_loss",
            greater_is_better=False, logging_steps=5, report_to="none", seed=config.seed,
            data_seed=config.seed, remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=arguments, train_dataset=training,
                          eval_dataset=validation, data_collator=CompletionCollator(tokenizer))
        result = trainer.train()
        evaluation = trainer.evaluate()
        model.save_pretrained(partial, safe_serialization=True)
        tokenizer.save_pretrained(partial)
        values = {
            "schemaVersion": 1, "version": version, "kind": "bootstrap",
            "baseModel": "Qwen/Qwen3-1.7B", "parentVersion": None,
            "dataset": _dataset_manifest(config.train),
            "training": {**asdict(config), "train": str(config.train),
                         "validation": str(config.validation), "output": str(config.output),
                         "trainLoss": result.training_loss, "evalLoss": evaluation.get("eval_loss")},
            "evaluation": {"status": "pending"},
            "createdAt": datetime.now(timezone.utc).isoformat(),
        }
        write_manifest(partial, values)
        os.rename(partial, config.output)
        return config.output
    except BaseException:
        # Deliberately retain the .partial directory for diagnosis; it can never be loaded.
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--validation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(train(BootstrapTrainingConfig(**vars(args))))


if __name__ == "__main__":
    main()
