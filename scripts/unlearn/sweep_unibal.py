"""Run reproducible UniBal parameter grids on TOFU or MUSE.

Examples:
  python scripts/unlearn/sweep_unibal.py tofu npo --dry-run
  python scripts/unlearn/sweep_unibal.py muse simnpo --cuda-devices 0
  python scripts/unlearn/sweep_unibal.py tofu all --eps 0.16 0.18 --eta 0.05
"""

import argparse
import hashlib
import itertools
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINERS = {"npo": "UniBalNPO", "simnpo": "UniBalSimNPO", "satimp": "UniBalSatImp"}

# Starting grids from the old UniBal scripts; incompatible implementation
# details (old DV weighting, uniform prototype mean, zero dual step) are excluded.
PROFILES = {
    ("tofu", "npo"): dict(
        splits=("forget01",),
        lr=("5e-5",),
        batch=(8,),
        accum=(2,),
        epochs=(15,),
        eps=(0.16, 0.18),
        eta=(0.05,),
        dv=(1.0, 2.0),
        beta=(0.1,),
        delta=(0.0,),
        gamma=(1.0,),
    ),
    ("tofu", "simnpo"): dict(
        splits=("forget01",),
        lr=("5e-5",),
        batch=(8,),
        accum=(4,),
        epochs=(10,),
        eps=(0.16, 0.18),
        eta=(0.05,),
        dv=(1.0, 2.0),
        beta=(3.5,),
        delta=(1.0,),
        gamma=(1.0,),
    ),
    ("tofu", "satimp"): dict(
        splits=("forget01",),
        lr=("5e-5",),
        batch=(8,),
        accum=(4,),
        epochs=(10,),
        eps=(0.16, 0.18),
        eta=(0.05,),
        dv=(1.0, 2.0),
        beta1=(5.0,),
        beta2=(1.0,),
        gamma=(1.0,),
    ),
    ("muse", "npo"): dict(
        splits=("Books",),
        lr=("5e-5",),
        batch=(4,),
        accum=(8,),
        epochs=(10,),
        eps=(0.14, 0.16, 0.18, 0.20),
        eta=(0.05, 0.1),
        dv=(0.5, 1.0),
        beta=(0.1,),
        delta=(0.0,),
        gamma=(1.0,),
    ),
    ("muse", "simnpo"): dict(
        splits=("Books",),
        lr=("5e-5",),
        batch=(4,),
        accum=(8,),
        epochs=(10,),
        eps=(0.14, 0.16, 0.18, 0.20),
        eta=(0.05, 0.1),
        dv=(0.5, 1.0, 2.0, 5.0, 10.0),
        beta=(4.5,),
        delta=(0.0,),
        gamma=(1.0,),
    ),
    ("muse", "satimp"): dict(
        splits=("News",),
        lr=("5e-5",),
        batch=(4,),
        accum=(4, 8),
        epochs=(10,),
        eps=(0.14, 0.16, 0.18),
        eta=(0.05, 0.1),
        dv=(5.0, 10.0),
        beta1=(5.0,),
        beta2=(1.0,),
        gamma=(1.0,),
    ),
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("tofu", "muse"))
    parser.add_argument("method", choices=(*TRAINERS, "all"))
    parser.add_argument(
        "--split", nargs="+", help="TOFU: forget01/05/10; MUSE: Books/News"
    )
    parser.add_argument("--lr", nargs="+", help="Learning rates")
    parser.add_argument("--batch", nargs="+", type=int)
    parser.add_argument("--accum", nargs="+", type=int)
    parser.add_argument("--epochs", nargs="+", type=int)
    parser.add_argument("--eps", nargs="+", type=float, help="Retain loss thresholds")
    parser.add_argument("--eta", nargs="+", type=float, help="Dual step sizes")
    parser.add_argument("--beta-dv", nargs="+", type=float, dest="dv")
    parser.add_argument("--beta", nargs="+", type=float, help="NPO/SimNPO beta")
    parser.add_argument("--delta", nargs="+", type=float, help="SimNPO delta")
    parser.add_argument("--gamma", nargs="+", type=float)
    parser.add_argument("--beta1", nargs="+", type=float, help="SatImp beta1")
    parser.add_argument("--beta2", nargs="+", type=float, help="SatImp beta2")
    parser.add_argument("--prototypes", type=int, default=40)
    parser.add_argument("--probe-batch", type=int, help="Default: TOFU 10, MUSE 2")
    parser.add_argument("--encoder", help="Semantic encoder checkpoint or local path")
    parser.add_argument("--medoid-sample-size", type=int, default=5000)
    parser.add_argument("--encoder-batch-size", type=int, default=32)
    parser.add_argument("--encoder-max-length", type=int, default=256)
    parser.add_argument("--model-path", help="Override the target model path")
    parser.add_argument("--model-config", help="Override the Hydra model config")
    parser.add_argument("--tokenizer-path", help="Override the tokenizer path")
    parser.add_argument(
        "--retain-logs-path", help="Reference evaluation log for a custom model"
    )
    parser.add_argument(
        "--python", default=sys.executable, help="Python for src/train.py"
    )
    parser.add_argument("--cuda-devices", help="CUDA_VISIBLE_DEVICES, e.g. 0 or 0,1")
    parser.add_argument(
        "--report-to", default="none", help="HF reporting integration, e.g. wandb"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "saves" / "unlearn")
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without training"
    )
    parser.add_argument(
        "--max-runs", type=int, help="Run/print only the first N combinations"
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--no-eval", action="store_true", help="Train without epoch or final evaluation"
    )
    args = parser.parse_args(argv)
    args.output_root = args.output_root.resolve()
    if any(
        value <= 0
        for value in (
            args.prototypes,
            args.medoid_sample_size,
            args.encoder_batch_size,
            args.encoder_max_length,
        )
    ) or (args.probe_batch is not None and args.probe_batch <= 0):
        parser.error("Prototype and encoder settings must be positive")
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")
    for name in ("batch", "accum", "epochs"):
        values = getattr(args, name)
        if values is not None and any(value <= 0 for value in values):
            parser.error(f"--{name} values must be positive")
    for name in ("eta", "dv", "beta", "gamma"):
        values = getattr(args, name)
        if values is not None and any(
            not math.isfinite(value) or value <= 0 for value in values
        ):
            parser.error(f"--{name} values must be positive")
    if args.eps is not None and any(
        not math.isfinite(value) or value < 0 for value in args.eps
    ):
        parser.error("--eps values must be nonnegative")
    if args.lr is not None:
        try:
            valid_lr = all(
                math.isfinite(float(value)) and float(value) > 0 for value in args.lr
            )
        except ValueError:
            valid_lr = False
        if not valid_lr:
            parser.error("--lr values must be positive numbers")
    if args.split is not None:
        if args.benchmark == "tofu":
            valid_splits = all(
                value.startswith("forget")
                and value[6:].isdigit()
                and 0 < int(value[6:]) < 100
                for value in args.split
            )
        else:
            valid_splits = all(value in ("Books", "News") for value in args.split)
        if not valid_splits:
            parser.error(
                "--split must be forget01/05/10-style on TOFU or Books/News on MUSE"
            )
    if args.model_path and not args.retain_logs_path and not args.no_eval:
        parser.error("--model-path requires --retain-logs-path when evaluation is on")
    return args


def fmt(value):
    return str(value).replace(".", "p")


def commands(args):
    methods = TRAINERS if args.method == "all" else (args.method,)
    for method in methods:
        profile = PROFILES[(args.benchmark, method)].copy()
        for key in profile:
            override = getattr(args, "split" if key == "splits" else key, None)
            if override is not None:
                profile[key] = tuple(override)
        keys = tuple(profile)
        for values in itertools.product(*(profile[key] for key in keys)):
            point = dict(zip(keys, values))
            split = point["splits"]
            suffix = "_".join(
                f"{key}{fmt(point[key])}" for key in keys if key != "splits"
            )
            if args.benchmark == "tofu":
                model = (
                    args.model_path or "open-unlearning/tofu_Llama-2-7b-chat-hf_full"
                )
                tokenizer = args.tokenizer_path or model
                model_config = args.model_config or "Llama-2-7b-chat-hf"
                number = split.removeprefix("forget")
                retain_split = f"retain{100 - int(number):02d}"
                holdout_split = f"holdout{number}"
                retain_logs = (
                    args.retain_logs_path
                    or f"./saves/eval/tofu_{model_config}_{retain_split}/TOFU_EVAL.json"
                )
                data_args = [
                    f"forget_split={split}",
                    f"retain_split={retain_split}",
                    f"holdout_split={holdout_split}",
                    f"eval.tofu.retain_logs_path={retain_logs}",
                ]
            else:
                model = args.model_path or f"muse-bench/MUSE-{split.lower()}_target"
                tokenizer = args.tokenizer_path or "meta-llama/Llama-2-7b-hf"
                model_config = args.model_config or (
                    "ICLM-7B" if split == "Books" else "Llama-2-7b-hf"
                )
                # The bundled Books log predates the ICLM-7B config name.
                default_log_model = (
                    "Llama-2-7b-hf"
                    if split == "Books" and model_config == "ICLM-7B"
                    else model_config
                )
                retain_logs = (
                    args.retain_logs_path
                    or f"./saves/eval/muse_{default_log_model}_{split}_retrain/MUSE_EVAL.json"
                )
                data_args = [
                    f"data_split={split}",
                    f"eval.muse.retain_logs_path={retain_logs}",
                ]
            probe_batch = args.probe_batch or (10 if args.benchmark == "tofu" else 2)
            fingerprint_input = (
                model,
                tokenizer,
                model_config,
                args.encoder,
                args.prototypes,
                probe_batch,
                args.medoid_sample_size,
                args.encoder_batch_size,
                args.encoder_max_length,
                args.retain_logs_path,
                args.no_eval,
            )
            fingerprint = hashlib.sha256(repr(fingerprint_input).encode()).hexdigest()[
                :8
            ]
            suffix += f"_k{args.prototypes}_cfg{fingerprint}"
            task_name = f"unibal_{args.benchmark}_{split}_{method}_{suffix}"
            output = (
                args.output_root / args.benchmark / split / TRAINERS[method] / suffix
            )
            command = [
                args.python,
                "src/train.py",
                "--config-name=unlearn.yaml",
                f"experiment=unlearn/{args.benchmark}/default",
                f"trainer={TRAINERS[method]}",
                f"model={model_config}",
                f"model.model_args.pretrained_model_name_or_path={model}",
                f"model.tokenizer_args.pretrained_model_name_or_path={tokenizer}",
                *data_args,
                f"task_name={task_name}",
                f"paths.output_dir={output}",
                "do_save=True",
                "trainer.args.ddp_find_unused_parameters=true",
                "trainer.args.gradient_checkpointing=true",
                f"trainer.args.report_to={args.report_to}",
                f"trainer.args.run_name={task_name}",
                "trainer.args.logging_steps=1",
                f"trainer.args.learning_rate={point['lr']}",
                f"trainer.args.per_device_train_batch_size={point['batch']}",
                f"trainer.args.gradient_accumulation_steps={point['accum']}",
                f"trainer.args.num_train_epochs={point['epochs']}",
                f"trainer.args.eval_strategy={'no' if args.no_eval else 'epoch'}",
                "trainer.args.eval_on_start=False",
                f"trainer.args.do_eval={str(not args.no_eval)}",
                f"trainer.method_args.beta_dv_forget={point['dv']}",
                f"trainer.method_args.retain_loss_eps={point['eps']}",
                f"trainer.method_args.dual_step_size={point['eta']}",
                f"trainer.method_args.n_prototypes={args.prototypes}",
                f"trainer.method_args.probe_batch_size={probe_batch}",
                f"trainer.method_args.medoid_sample_size={args.medoid_sample_size}",
                f"trainer.method_args.encoder_batch_size={args.encoder_batch_size}",
                f"trainer.method_args.encoder_max_length={args.encoder_max_length}",
                f"trainer.method_args.gamma={point['gamma']}",
            ]
            if args.encoder:
                command.append(
                    f"trainer.method_args.semantic_encoder_name={args.encoder}"
                )
            if method in ("npo", "simnpo"):
                command.append(f"trainer.method_args.beta={point['beta']}")
            if method == "simnpo":
                command.append(f"trainer.method_args.delta={point['delta']}")
            if method == "satimp":
                command.extend(
                    (
                        f"trainer.method_args.beta1={point['beta1']}",
                        f"trainer.method_args.beta2={point['beta2']}",
                    )
                )
            yield task_name, command


def main(argv=None):
    args = parse_args(argv)
    if not args.dry_run and not (ROOT / "src" / "train.py").is_file():
        raise SystemExit(f"Cannot find {ROOT / 'src' / 'train.py'}")
    environment = os.environ.copy()
    if args.cuda_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    environment.setdefault("WANDB_PROJECT", "UniBal")
    count = 0
    failures = 0
    for task_name, command in commands(args):
        if args.max_runs is not None and count >= args.max_runs:
            break
        count += 1
        print(f"[{count}] {task_name}", flush=True)
        if args.dry_run:
            print(
                subprocess.list2cmdline(command)
                if os.name == "nt"
                else shlex.join(command),
                flush=True,
            )
            continue
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode:
            failures += 1
            print(
                f"FAILED ({result.returncode}): {task_name}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                return result.returncode
    print(f"Sweep finished: {count} combinations, {failures} failures.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
