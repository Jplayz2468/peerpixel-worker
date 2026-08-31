"""Optional, owner-operated prompt-adapter preference training.

Nothing imports this module from the render path. A coordinator integration
may construct :class:`TrainerCapability` and call ``poll_when_idle`` between
jobs; the capability remains inert unless the owner explicitly enables it.
"""
from __future__ import annotations

import hashlib
import base64
import io
import json
import math
import os
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from .dpo import weighted_dpo_loss
from .lora_manifest import load_manifest, safe_version, write_manifest
from . import api, config


MAX_SNAPSHOT_BYTES = 128 * 1024 * 1024
MAX_LEARNING_RATE = 1e-5
MAX_TRAINING_STEPS = 10_000
SHA256 = re.compile(r"^[0-9a-f]{64}$")
EVALUATION_GATES = (
    "responseStructure", "weightedPreferenceAccuracy", "promptFidelity",
    "diversity", "phraseConcentration", "anchorDrift",
)


class TrainingError(RuntimeError):
    """A stable failure reason safe to report to the coordinator."""


@dataclass(frozen=True)
class TrainingConfig:
    beta: float
    max_steps: int
    learning_rate: float
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    seed: int = 2468

    @classmethod
    def from_payload(cls, value: Mapping) -> "TrainingConfig":
        if not isinstance(value, Mapping):
            raise ValueError("training configuration must be an object")
        try:
            config = cls(
                beta=float(value["beta"]),
                max_steps=int(value["maxSteps"]),
                learning_rate=float(value["learningRate"]),
                batch_size=int(value.get("batchSize", 1)),
                gradient_accumulation_steps=int(
                    value.get("gradientAccumulationSteps", 8)),
                seed=int(value.get("seed", 2468)),
            )
        except (KeyError, TypeError, ValueError):
            raise ValueError("invalid training configuration") from None
        if not math.isfinite(config.beta) or config.beta <= 0:
            raise ValueError("training beta must be positive and finite")
        if config.max_steps <= 0 or config.max_steps > MAX_TRAINING_STEPS:
            raise ValueError("training maxSteps is outside the worker bound")
        if (not math.isfinite(config.learning_rate)
                or config.learning_rate <= 0
                or config.learning_rate > MAX_LEARNING_RATE):
            raise ValueError("training learningRate is outside the worker bound")
        if config.batch_size <= 0 or config.batch_size > 32:
            raise ValueError("training batchSize is outside the worker bound")
        if (config.gradient_accumulation_steps <= 0
                or config.gradient_accumulation_steps > 128):
            raise ValueError(
                "training gradientAccumulationSteps is outside the worker bound")
        if config.seed < 0 or config.seed >= 2 ** 63:
            raise ValueError("training seed is outside the worker bound")
        return config

    def manifest(self) -> dict:
        return {
            "beta": self.beta,
            "maxSteps": self.max_steps,
            "learningRate": self.learning_rate,
            "batchSize": self.batch_size,
            "gradientAccumulationSteps": self.gradient_accumulation_steps,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TrainingLease:
    run_id: str
    lease_token: str
    device_id: str
    expires_at: float
    snapshot_digest: str
    parent_version: str
    candidate_version: str
    training: TrainingConfig
    snapshot_body: bytes = field(default=b"", repr=False)
    active_adapter: Path | None = None
    staging_root: Path | None = None

    @classmethod
    def from_payload(cls, value: Mapping, *, expected_device_id: str,
                     now: float) -> "TrainingLease":
        if not isinstance(value, Mapping):
            raise ValueError("training lease must be an object")
        try:
            lease = cls(
                run_id=safe_version(str(value["runId"])),
                lease_token=str(value["leaseToken"]),
                device_id=str(value["deviceId"]),
                expires_at=float(value["expiresAt"]),
                snapshot_digest=str(value["snapshotDigest"]).lower(),
                parent_version=safe_version(str(value["parentVersion"])),
                candidate_version=safe_version(str(value["candidateVersion"])),
                training=TrainingConfig.from_payload(value["training"]),
            )
        except (KeyError, TypeError):
            raise ValueError("invalid training lease") from None
        if not lease.lease_token:
            raise ValueError("training lease is missing its signed token")
        if lease.device_id != str(expected_device_id):
            raise ValueError("training lease belongs to another device")
        if not math.isfinite(lease.expires_at) or lease.expires_at <= float(now):
            raise ValueError("training lease has expired")
        if not SHA256.fullmatch(lease.snapshot_digest):
            raise ValueError("training lease has an invalid snapshot digest")
        if lease.candidate_version == lease.parent_version:
            raise ValueError("candidate version must differ from its parent")
        return lease

    def bind_local(self, snapshot_body: bytes, *, active_adapter: Path,
                   staging_root: Path) -> "TrainingLease":
        body = bytes(snapshot_body)
        if not body or len(body) > MAX_SNAPSHOT_BYTES:
            raise TrainingError("snapshot_size_invalid")
        if hashlib.sha256(body).hexdigest() != self.snapshot_digest:
            raise TrainingError("snapshot_digest_mismatch")
        return replace(
            self,
            snapshot_body=body,
            active_adapter=Path(active_adapter),
            staging_root=Path(staging_root),
        )


@dataclass(frozen=True)
class TrainingReport:
    run_id: str
    status: str
    artifact_path: Path | None = None
    artifact_digest: str | None = None
    metrics: Mapping = field(default_factory=dict)
    reason: str | None = None

    @classmethod
    def failed(cls, run_id: str, reason: str) -> "TrainingReport":
        return cls(run_id=run_id, status="failed", reason=reason)

    def payload(self) -> dict:
        value = {
            "runId": self.run_id,
            "status": self.status,
            "metrics": dict(self.metrics),
        }
        if self.artifact_digest:
            value["artifactDigest"] = self.artifact_digest
        if self.reason:
            value["reason"] = self.reason
        return value


def _preference_rows(rows, *, allow_empty: bool) -> list[dict]:
    if not isinstance(rows, list) or (not rows and not allow_empty):
        raise TrainingError("snapshot_records_invalid")
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise TrainingError("snapshot_records_invalid")
        if any(not isinstance(row.get(name), str) or not row[name].strip()
               for name in ("prompt", "chosen", "rejected")):
            raise TrainingError("snapshot_records_invalid")
        try:
            weight = float(row.get("weight"))
        except (TypeError, ValueError):
            raise TrainingError("snapshot_weight_invalid") from None
        if not math.isfinite(weight) or weight <= 0:
            raise TrainingError("snapshot_weight_invalid")
        normalized.append({**row, "prompt": row["prompt"].strip(),
                           "chosen": row["chosen"].strip(),
                           "rejected": row["rejected"].strip(), "weight": weight})
    return normalized


@dataclass(frozen=True)
class PreferenceSnapshot:
    train: list[dict]
    replay: list[dict]
    evaluation: list[dict]

    @property
    def training(self) -> list[dict]:
        return [*self.train, *self.replay]


def _snapshot_rows(body: bytes) -> PreferenceSnapshot:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise TrainingError("snapshot_json_invalid") from None
    if not isinstance(value, dict):
        raise TrainingError("snapshot_records_invalid")
    snapshot = PreferenceSnapshot(
        train=_preference_rows(value.get("train"), allow_empty=False),
        replay=_preference_rows(value.get("replay"), allow_empty=True),
        evaluation=_preference_rows(value.get("evaluation"), allow_empty=True),
    )
    training_events = {row.get("eventId") for row in snapshot.training if row.get("eventId")}
    evaluation_events = {row.get("eventId") for row in snapshot.evaluation if row.get("eventId")}
    if training_events & evaluation_events:
        raise TrainingError("snapshot_split_overlap")
    return snapshot


def _smoke_evaluation(metrics: Mapping) -> dict:
    loss = metrics.get("trainLoss")
    steps = metrics.get("steps")
    numeric_metrics_finite = all(
        not isinstance(value, (int, float))
        or (not isinstance(value, bool) and math.isfinite(float(value)))
        for value in metrics.values()
    )
    passed = (numeric_metrics_finite
              and isinstance(loss, (int, float)) and not isinstance(loss, bool)
              and math.isfinite(float(loss))
              and isinstance(steps, int) and not isinstance(steps, bool) and steps > 0)
    return {
        "mode": "smoke",
        "passed": passed,
        "gates": {name: passed for name in EVALUATION_GATES},
        "metrics": dict(metrics),
    }


def _is_inside(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def run_training(lease: TrainingLease, *, train_backend: Callable | None = None,
                 progress: Callable | None = None,
                 created_at: str | None = None) -> TrainingReport:
    """Train one lease into an inactive, atomically staged candidate.

    The active adapter is read and validated but never written. The only
    publish performed here is ``*.partial`` to a sibling candidate directory;
    changing the runtime adapter remains a coordinator promotion operation.
    """
    if not lease.snapshot_body or lease.active_adapter is None or lease.staging_root is None:
        raise TrainingError("training_lease_not_bound")
    if hashlib.sha256(lease.snapshot_body).hexdigest() != lease.snapshot_digest:
        raise TrainingError("snapshot_digest_mismatch")
    snapshot = _snapshot_rows(lease.snapshot_body)

    active = lease.active_adapter.resolve()
    manifest = load_manifest(active / "manifest.json")
    if manifest["version"] != lease.parent_version:
        raise TrainingError("parent_adapter_mismatch")

    staging = lease.staging_root.resolve()
    partial = staging / f"{lease.candidate_version}.partial"
    candidate = staging / lease.candidate_version
    if _is_inside(partial, active) or _is_inside(candidate, active):
        raise TrainingError("staging_overlaps_active_adapter")
    if candidate.exists():
        raise FileExistsError("refusing to overwrite a trainer artifact")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    backend = train_backend or _train_with_trl
    metrics = backend(lease, snapshot.training, snapshot.evaluation, partial) if train_backend else backend(
        lease, snapshot.training, snapshot.evaluation, partial, progress=progress)
    if metrics is None:
        metrics = {}
    if not isinstance(metrics, Mapping):
        raise TrainingError("training_metrics_invalid")
    evaluation = _smoke_evaluation(metrics)
    values = {
        "schemaVersion": 1,
        "complete": True,
        "version": lease.candidate_version,
        "kind": "preference",
        "baseModel": manifest["baseModel"],
        "parentVersion": lease.parent_version,
        "snapshotDigest": lease.snapshot_digest,
        "dataset": {
            "snapshotDigest": lease.snapshot_digest,
            "trainRecords": len(snapshot.train),
            "replayRecords": len(snapshot.replay),
            "evaluationRecords": len(snapshot.evaluation),
            "preferenceRecords": len(snapshot.training),
            "preferenceWeight": sum(row["weight"] for row in snapshot.training),
        },
        "training": lease.training.manifest(),
        "evaluation": evaluation,
        "createdAt": created_at or datetime.now(timezone.utc).isoformat(),
    }
    write_manifest(partial, values)
    os.rename(partial, candidate)
    artifact_digest = hashlib.sha256(package_candidate(candidate)).hexdigest()
    return TrainingReport(
        run_id=lease.run_id,
        status="staged",
        artifact_path=candidate,
        artifact_digest=artifact_digest,
        metrics=dict(metrics),
    )


def _train_with_trl(lease: TrainingLease, rows: list[dict], _evaluation_rows: list[dict],
                    output_dir: Path, progress: Callable | None = None) -> dict:
    """Run the optional GPU backend, importing its dependencies only on use."""
    try:
        import torch
        from datasets import Dataset
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import DPOConfig, DPOTrainer
        from trl.trainer.dpo_trainer import DataCollatorForPreference
    except ImportError as error:
        raise TrainingError("trainer_dependencies_missing") from error

    from . import model_hub

    base_path = model_hub.ensure("qwen3-1.7b")
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_options = {"local_files_only": True}
    use_cuda = torch.cuda.is_available()
    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    if use_cuda:
        model_options.update({
            "device_map": "auto",
            "dtype": torch.bfloat16 if use_bf16 else torch.float16,
        })
    base = AutoModelForCausalLM.from_pretrained(base_path, **model_options)
    model = PeftModel.from_pretrained(
        base, str(lease.active_adapter), is_trainable=True,
        local_files_only=True,
    )
    model.set_adapter("default")
    model.config.use_cache = False

    class WeightedPreferenceCollator(DataCollatorForPreference):
        def torch_call(self, examples):
            batch = super().torch_call(examples)
            batch["preference_weight"] = torch.tensor(
                [float(example["weight"]) for example in examples],
                dtype=torch.float32,
            )
            return batch

    class WeightedDPOTrainer(DPOTrainer):
        _peerpixel_weight = None

        def compute_loss(self, model, inputs, return_outputs=False,
                         num_items_in_batch=None):
            self._peerpixel_weight = inputs.get("preference_weight")
            if self._peerpixel_weight is None:
                raise ValueError("preference weight was lost before DPO reduction")
            try:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )
            finally:
                self._peerpixel_weight = None

        def dpo_loss(self, chosen_logps, rejected_logps, ref_chosen_logps,
                     ref_rejected_logps, loss_type="sigmoid", model_output=None):
            if loss_type != "sigmoid":
                raise ValueError("PeerPixel preference training uses sigmoid DPO")
            weight = self._peerpixel_weight.to(
                device=chosen_logps.device, dtype=chosen_logps.dtype)
            loss = weighted_dpo_loss(
                chosen_logps, rejected_logps, ref_chosen_logps,
                ref_rejected_logps, self.beta, weight,
            )
            chosen_rewards = self.beta * (chosen_logps - ref_chosen_logps).detach()
            rejected_rewards = self.beta * (rejected_logps - ref_rejected_logps).detach()
            return loss, chosen_rewards, rejected_rewards

    dataset = Dataset.from_list([{
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
        "weight": row["weight"],
    } for row in rows])
    arguments = DPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        beta=lease.training.beta,
        max_steps=lease.training.max_steps,
        learning_rate=lease.training.learning_rate,
        per_device_train_batch_size=lease.training.batch_size,
        gradient_accumulation_steps=lease.training.gradient_accumulation_steps,
        bf16=use_bf16,
        fp16=bool(use_cuda and not use_bf16),
        gradient_checkpointing=True,
        max_length=512,
        loss_type="sigmoid",
        save_strategy="no",
        eval_strategy="no",
        logging_steps=5,
        report_to="none",
        remove_unused_columns=False,
        seed=lease.training.seed,
        data_seed=lease.training.seed,
    )
    trainer = WeightedDPOTrainer(
        model=model,
        ref_model=None,
        args=arguments,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=WeightedPreferenceCollator(
            pad_token_id=tokenizer.pad_token_id),
        callbacks=[type("ProgressCallback", (TrainerCallback,), {
            "on_step_end": lambda self, args, state, control, **kwargs: (
                progress(int(state.global_step), int(state.max_steps))
                if progress and (int(state.global_step) % 5 == 0 or state.global_step == state.max_steps) else None
            ) or control,
        })()],
    )
    result = trainer.train()
    model.set_adapter("default")
    model.save_pretrained(
        output_dir, selected_adapters=["default"], safe_serialization=True)
    return {
        "trainLoss": float(result.training_loss),
        "steps": int(result.global_step),
    }


def package_candidate(candidate: Path) -> bytes:
    """Return stable ZIP bytes for one complete staged adapter directory."""
    candidate = Path(candidate)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as archive:
        for path in sorted(candidate.rglob("*"), key=lambda item: item.relative_to(candidate).as_posix()):
            if path.is_symlink():
                raise TrainingError("artifact_symlink_invalid")
            if not path.is_file():
                continue
            name = path.relative_to(candidate).as_posix()
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())
    artifact = output.getvalue()
    if not artifact or len(artifact) > 64 * 1024 * 1024:
        raise TrainingError("artifact_size_invalid")
    return artifact


class TrainingHttpClient:
    """Authenticated binding for the coordinator's trainer endpoints."""

    def lease_training(self, device_id: str):
        return api._call("/api/worker/training/lease", method="POST",
                         payload={"deviceId": str(device_id)}, timeout=30)

    def download_snapshot(self, lease: TrainingLease) -> bytes:
        token = config.read().get("token", "")
        request = urllib.request.Request(
            f"{config.API}/api/worker/training/snapshot/{lease.run_id}",
            headers={
                "user-agent": api.USER_AGENT,
                "authorization": f"Bearer {token}",
                "x-peerpixel-device": lease.device_id,
                "x-peerpixel-training-lease": lease.lease_token,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(MAX_SNAPSHOT_BYTES + 1)
        except urllib.error.HTTPError as error:
            raise api.ApiError(error.code, "training_snapshot_failed") from None

    def upload_candidate(self, lease: TrainingLease, artifact_path: Path,
                         report: TrainingReport) -> dict:
        artifact = package_candidate(artifact_path)
        digest = hashlib.sha256(artifact).hexdigest()
        if report.artifact_digest and report.artifact_digest != digest:
            raise TrainingError("artifact_digest_mismatch")
        manifest = json.loads((Path(artifact_path) / "manifest.json").read_text(encoding="utf-8"))
        for attempt in range(3):
            try:
                token = config.read().get("token", "")
                headers = {
                    "user-agent": api.USER_AGENT,
                    "authorization": f"Bearer {token}",
                    "content-type": "application/zip",
                    "x-peerpixel-device": lease.device_id,
                    "x-peerpixel-training-lease": lease.lease_token,
                    "x-peerpixel-snapshot-digest": lease.snapshot_digest,
                    "x-peerpixel-artifact-digest": digest,
                    "x-peerpixel-manifest": base64.b64encode(json.dumps(manifest, separators=(",", ":")).encode()).decode(),
                    "x-peerpixel-evaluation": base64.b64encode(json.dumps(manifest["evaluation"], separators=(",", ":")).encode()).decode(),
                }
                request = urllib.request.Request(
                    f"{config.API}/api/worker/training/candidate/{lease.run_id}",
                    data=artifact, headers=headers, method="PUT")
                with urllib.request.urlopen(request, timeout=900) as response:
                    return json.loads(response.read().decode() or "{}")
            except urllib.error.HTTPError as raw_error:
                try: body = json.loads(raw_error.read().decode() or "{}")
                except Exception: body = {}
                error = api.ApiError(raw_error.code, body.get("error", "candidate_upload_failed"), body)
                if error.status == 422 and error.body.get("accepted"):
                    return error.body
                if error.status < 500 or attempt == 2:
                    raise error
                time.sleep(2 ** attempt)
            except api.ApiError as error:
                # Validation failure is a successfully stored non-promoting rehearsal.
                if error.status == 422 and error.body.get("accepted"):
                    return error.body
                if error.status < 500 or attempt == 2:
                    raise
                time.sleep(2 ** attempt)
        raise TrainingError("candidate_upload_failed")

    def report_training(self, lease: TrainingLease, report: TrainingReport) -> dict:
        return api._call(
            f"/api/worker/training/report/{lease.run_id}", method="POST",
            payload={
                **report.payload(),
                "deviceId": lease.device_id,
                "leaseToken": lease.lease_token,
                "snapshotDigest": lease.snapshot_digest,
            }, timeout=120)

    def report_progress(self, lease: TrainingLease, step: int, total: int, eta_seconds: float) -> dict:
        return api._call(f"/api/worker/training/progress/{lease.run_id}", method="POST", payload={
            "deviceId": lease.device_id, "leaseToken": lease.lease_token,
            "step": step, "total": total, "etaSeconds": max(0, round(eta_seconds)),
        }, timeout=30)


class TrainerCapability:
    """A disabled-by-default lease poller for an explicitly opted-in owner PC.

    ``client`` is the narrow coordinator seam. It supplies ``lease_training``,
    ``download_snapshot``, ``upload_candidate``, and ``report_training``. This
    keeps the trainer independent of the render websocket and of an endpoint
    payload that belongs to the coordinator repository.
    """

    def __init__(self, client, *, device_id: str, enabled: bool = False,
                 is_idle: Callable[[], bool] | None = None,
                 active_adapter: Path | Callable[[], Path] | None = None,
                 staging_root: Path | None = None,
                 train_backend: Callable | None = None,
                 on_promoted: Callable[[Path, Mapping], None] | None = None,
                 on_parent_selected: Callable[[Path], None] | None = None,
                 on_training_start: Callable[[], None] | None = None,
                 on_training_end: Callable[[], None] | None = None,
                 now: Callable[[], float] | None = None):
        self.client = client
        self.device_id = str(device_id)
        self.enabled = bool(enabled)
        self.is_idle = is_idle or (lambda: True)
        self.active_adapter = active_adapter
        self.staging_root = Path(staging_root) if staging_root is not None else None
        self.train_backend = train_backend
        self.on_promoted = on_promoted
        self.on_parent_selected = on_parent_selected
        self.on_training_start = on_training_start
        self.on_training_end = on_training_end
        self.now = now or __import__("time").time
        self.last_report: TrainingReport | None = None
        self._training = False
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> tuple[str, ...]:
        return ("train",) if self.enabled else ()

    @property
    def training(self) -> bool:
        with self._lock:
            return self._training

    def _active_path(self) -> Path:
        value = self.active_adapter() if callable(self.active_adapter) else self.active_adapter
        if value is None:
            raise TrainingError("active_adapter_unavailable")
        return Path(value)

    def _parent_path(self, lease: TrainingLease) -> Path:
        active = self._active_path().resolve()
        try:
            if load_manifest(active / "manifest.json")["version"] == lease.parent_version:
                return active
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if self.staging_root is None:
            raise TrainingError("trainer_staging_unavailable")
        retained = (self.staging_root.resolve() / lease.parent_version).resolve()
        try:
            manifest = load_manifest(retained / "manifest.json")
        except (OSError, ValueError, KeyError, TypeError):
            raise TrainingError("parent_adapter_mismatch") from None
        if manifest["version"] != lease.parent_version:
            raise TrainingError("parent_adapter_mismatch")
        if self.on_parent_selected is not None:
            self.on_parent_selected(retained)
        return retained

    def _failure(self, lease: TrainingLease, reason: str) -> None:
        report = TrainingReport.failed(lease.run_id, reason)
        self.last_report = report
        try:
            self.client.report_training(lease, report)
        except Exception:
            pass  # A short lease makes this retry-safe on the next coordinator run.

    def poll_when_idle(self) -> bool:
        """Handle at most one lease, returning whether one was received."""
        if not self.enabled or not self.is_idle():
            return False
        with self._lock:
            if self._training:
                return False
            self._training = True

        lease = None
        training_started = False
        try:
            payload = self.client.lease_training(self.device_id)
            if not payload:
                return False
            lease = TrainingLease.from_payload(
                payload, expected_device_id=self.device_id, now=self.now())
            if self.staging_root is None:
                raise TrainingError("trainer_staging_unavailable")
            body = self.client.download_snapshot(lease)
            lease = lease.bind_local(
                body, active_adapter=self._parent_path(lease),
                staging_root=self.staging_root,
            )
            if self.on_training_start is not None:
                self.on_training_start()
            training_started = True
            started = self.now()
            def progress(step, total):
                elapsed = max(0.001, self.now() - started)
                eta = (elapsed / max(1, step)) * max(0, total - step)
                try:
                    self.client.report_progress(lease, step, total, eta)
                except Exception:
                    pass
            if self.train_backend is None:
                from .training_process import run_training_isolated
                timeout = max(1.0, lease.expires_at - self.now() - 15.0)
                report = run_training_isolated(
                    lease, timeout=timeout, progress=progress)
            else:
                report = run_training(
                    lease, train_backend=self.train_backend, progress=progress)
            response = self.client.upload_candidate(lease, report.artifact_path, report) or {}
            if response.get("promoted") and self.on_promoted is not None:
                self.on_promoted(report.artifact_path, response)
            self.client.report_training(lease, report)
            self.last_report = report
            return True
        except KeyboardInterrupt:
            if lease is not None:
                self._failure(lease, "interrupted")
            raise
        except Exception as error:
            if lease is not None:
                if isinstance(error, TrainingError):
                    reason = str(error)
                else:
                    detail = " ".join(str(error).split())[:120]
                    reason = f"{type(error).__name__}:{detail}" if detail else type(error).__name__
                self._failure(lease, reason)
            return payload is not None if "payload" in locals() else False
        finally:
            if training_started and self.on_training_end is not None:
                try:
                    self.on_training_end()
                except Exception:
                    pass
            with self._lock:
                self._training = False
