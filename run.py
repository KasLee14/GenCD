# run.py
import argparse
import ast
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model.trainer import GenCD


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenCD on a prepared dataset.")
    parser.add_argument("--dataset-dir", type=str, default=None)
    parser.add_argument("--output-file", type=str, default=None)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--best-model-path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embedding-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-block", type=int, default=None)
    parser.add_argument("--dropout-ratio", type=float, default=None)
    parser.add_argument("--summary-json", type=str, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None)
    parser.add_argument("--resume-checkpoint", action="store_true")
    parser.add_argument("--resume-from-best", action="store_true")
    parser.add_argument("--resume-start-epoch", type=int, default=None)
    parser.add_argument("--resume-best-epoch", type=int, default=None)
    parser.add_argument("--resume-best-gauc", type=float, default=None)
    parser.add_argument("--resume-no-improve-epochs", type=int, default=None)
    return parser.parse_args()


def normalize_dataset_dir(path_text):
    dataset_path = Path(path_text)
    dataset_str = str(dataset_path)
    if not dataset_str.endswith(("/", "\\")):
        dataset_str += "/"
    return dataset_str


def apply_overrides(args):
    if args.dataset_dir is not None:
        Config.dataset_adr = normalize_dataset_dir(args.dataset_dir)
    if args.output_file is not None:
        Config.output_file = args.output_file
    if args.model_path is not None:
        Config.model_path = args.model_path
    if args.best_model_path is not None:
        Config.best_model_path = args.best_model_path
    if args.epochs is not None:
        Config.epochs = args.epochs
    if args.batch_size is not None:
        Config.batch_size = args.batch_size
    if args.lr is not None:
        Config.lr = args.lr
    if args.device is not None:
        Config.device = torch.device(args.device)
    if args.embedding_size is not None:
        Config.embedding_size = args.embedding_size
    if args.max_seq_len is not None:
        Config.max_seq_len = args.max_seq_len
    if args.num_layers is not None:
        Config.num_layers = args.num_layers
    if args.num_block is not None:
        Config.num_block = args.num_block
    if args.dropout_ratio is not None:
        Config.dropout_ratio = args.dropout_ratio
    if args.summary_json is not None:
        Config.summary_json = args.summary_json
    if args.checkpoint_path is not None:
        Config.checkpoint_path = args.checkpoint_path
    Config.resume_checkpoint = bool(args.resume_checkpoint)
    Config.resume_from_best = bool(args.resume_from_best)
    if args.resume_start_epoch is not None:
        Config.resume_start_epoch = args.resume_start_epoch
    if args.resume_best_epoch is not None:
        Config.resume_best_epoch = args.resume_best_epoch
    if args.resume_best_gauc is not None:
        Config.resume_best_gauc = args.resume_best_gauc
    if args.resume_no_improve_epochs is not None:
        Config.resume_no_improve_epochs = args.resume_no_improve_epochs


def resolve_time_column(df):
    if "timestamp" in df.columns:
        return "timestamp"
    if len(df.columns) >= 4:
        return df.columns[3]
    return None


def preprocess_time(series):
    return pd.to_datetime(series).astype("int64") // 10**9


def prepare_interaction_frame(df):
    time_col = resolve_time_column(df)
    frame = df.copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)

    if time_col is None:
        frame["timestamp"] = 0.0
        ordered = frame.sort_values(["user_id", "_row_order"], kind="mergesort")
        return ordered.reset_index(drop=True), False

    frame["timestamp"] = preprocess_time(frame[time_col])
    ordered = frame.sort_values(["user_id", "timestamp", "_row_order"], kind="mergesort")
    return ordered.reset_index(drop=True), True


RESULT_HEADER = (
    "dataset\tmodel_name\tepochs\tauc\tgauc\tacc\trmse\tmae"
    "\tnum_layers\tnum_block\tlambda_cl\tlambda_dmr\tmask_rate\tmax_seq_len\tdropout_ratio"
)
RESULT_COLUMN_COUNT = len(RESULT_HEADER.split("\t"))
LEGACY_RESULT_HEADERS = {
    "dataset\tmodel_name\tepochs\tauc\tgauc\tacc\trmse\tmae",
    "dataset\tmodel_name\tepochs\tauc\tacc\trmse\tmae",
}


def resolve_result_context(dataset_dir):
    split_report_path = dataset_dir / "split_report.json"
    if split_report_path.exists():
        report = json.loads(split_report_path.read_text(encoding="utf-8"))
        dataset_name = report.get("dataset", dataset_dir.name)
        if "split_ratio" in report:
            split_ratio = report["split_ratio"]
            split_line = f"# split_method: dataset={dataset_name}; prepared dataset train:test={split_ratio}"
        else:
            split_ratio = report.get("method", {}).get(
                "per_user_time_order_split",
                "8:2",
            )
            split_line = (
                f"# split_method: dataset={dataset_name}; "
                f"merged train/valid/test, dropped valid, per-user time-order train:test={split_ratio}"
            )
        return dataset_name, split_line

    dataset_name = dataset_dir.name
    split_line = f"# split_method: dataset={dataset_name}; original train/test"
    return dataset_name, split_line


def format_metric(value):
    return "nan" if np.isnan(value) else f"{float(value):.6f}"


def format_hparam(value):
    if value is None:
        return "na"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):g}"
    return str(value)


def result_hparams(config):
    return {
        "embedding_size": getattr(config, "embedding_size", None),
        "num_layers": getattr(config, "num_layers", None),
        "num_block": getattr(config, "num_block", None),
        "lambda_cl": getattr(config, "lambda_cl", None),
        "lambda_dmr": getattr(config, "lambda_dmr", None),
        "mask_rate": getattr(config, "mask_rate", None),
        "max_seq_len": getattr(config, "max_seq_len", None),
        "dropout_ratio": getattr(config, "dropout_ratio", None),
    }


def split_spec_slug(split_line):
    split_text = split_line
    if split_text.startswith("# split_method:"):
        split_text = split_text.split(":", 1)[1].strip()

    slug_chars = []
    for char in split_text:
        if char.isalnum():
            slug_chars.append(char.lower())
        else:
            slug_chars.append("_")

    slug = "".join(slug_chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def to_builtin(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def write_summary(summary_json_path, payload):
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps(to_builtin(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def normalize_result_text(existing_text):
    normalized_lines = [RESULT_HEADER]

    for raw_line in existing_text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            if normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        if stripped == RESULT_HEADER or stripped in LEGACY_RESULT_HEADERS:
            continue
        if stripped.startswith("#"):
            normalized_lines.append(stripped)
            continue

        parts = stripped.split("\t")
        if len(parts) == 7:
            parts.insert(4, "nan")
        if len(parts) == 8:
            parts.extend(["na"] * (RESULT_COLUMN_COUNT - 8))
        elif len(parts) < RESULT_COLUMN_COUNT:
            parts.extend(["na"] * (RESULT_COLUMN_COUNT - len(parts)))
        normalized_lines.append("\t".join(parts[:RESULT_COLUMN_COUNT]))

    while len(normalized_lines) > 1 and normalized_lines[-1] == "":
        normalized_lines.pop()
    return "\n".join(normalized_lines) + "\n"


def write_result(output_file, dataset_name, split_line, model_name, epochs, metrics, hyperparams):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    existing_text = output_file.read_text(encoding="utf-8") if output_file.exists() else ""
    normalized_text = normalize_result_text(existing_text)
    if existing_text != normalized_text:
        output_file.write_text(normalized_text, encoding="utf-8")
    existing_text = normalized_text

    non_empty_lines = [line.strip() for line in existing_text.splitlines() if line.strip()]
    last_split_line = next(
        (line for line in reversed(non_empty_lines) if line.startswith("# split_method:")),
        None,
    )

    with output_file.open("a", encoding="utf-8") as f:
        if last_split_line != split_line:
            if last_split_line is not None and not existing_text.endswith("\n\n"):
                f.write("\n")
            f.write(split_line + "\n")
        f.write(
            f"{dataset_name}\t{model_name}\t{epochs}\t"
            f"{format_metric(metrics['auc'])}\t{format_metric(metrics['gauc'])}\t"
            f"{format_metric(metrics['acc'])}\t{format_metric(metrics['rmse'])}\t"
            f"{format_metric(metrics['mae'])}\t"
            f"{format_hparam(hyperparams['num_layers'])}\t"
            f"{format_hparam(hyperparams['num_block'])}\t"
            f"{format_hparam(hyperparams['lambda_cl'])}\t"
            f"{format_hparam(hyperparams['lambda_dmr'])}\t"
            f"{format_hparam(hyperparams['mask_rate'])}\t"
            f"{format_hparam(hyperparams['max_seq_len'])}\t"
            f"{format_hparam(hyperparams['dropout_ratio'])}\n"
        )


def select_gauc_user_ids(frames, min_interactions_exclusive=100):
    valid_frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not valid_frames:
        return set()

    user_counts = (
        pd.concat([frame[["user_id"]] for frame in valid_frames], ignore_index=True)
        .groupby("user_id")
        .size()
        .reset_index(name="interaction_count")
    )
    selected = user_counts[user_counts["interaction_count"] > int(min_interactions_exclusive)]
    return set(selected["user_id"].astype(int).tolist())


run_start = time.perf_counter()

args = parse_args()
apply_overrides(args)
dataset_dir = Path(Config.dataset_adr)
dataset_name, split_line = resolve_result_context(dataset_dir)
output_file = Path(getattr(Config, "output_file", "gencd_result.txt"))
Config.dataset_name = dataset_name
Config.run_label = dataset_name
logging.getLogger().setLevel(logging.INFO)

train_data = pd.read_csv(dataset_dir / "train.csv")
test_data = pd.read_csv(dataset_dir / "test.csv")
df_item = pd.read_csv(dataset_dir / "item.csv")

train_data, train_has_timestamp = prepare_interaction_frame(train_data)
test_data, test_has_timestamp = prepare_interaction_frame(test_data)

if train_has_timestamp != test_has_timestamp:
    raise ValueError(
        "Inconsistent timestamp availability across splits. "
        f"train.csv has_timestamp={train_has_timestamp}, test.csv has_timestamp={test_has_timestamp}."
    )

Config.use_time_encoding = bool(train_has_timestamp)
Config.dataset_has_timestamp = bool(train_has_timestamp)

if Config.use_time_encoding:
    logging.info("Detected timestamps in dataset; GCE temporal bias is enabled.")
else:
    logging.info(
        "No timestamp column detected; GCE temporal bias is disabled and "
        "per-user sequence order falls back to the original file order."
    )

item2knowledge = {}
knowledge_set = set()
for _, s in df_item.iterrows():
    item_id = int(s["item_id"])
    knowledge_codes = np.asarray(sorted(set(ast.literal_eval(s["knowledge_code"]))), dtype=np.int64)
    item2knowledge[item_id] = knowledge_codes
    knowledge_set.update(knowledge_codes.tolist())

user_n = np.max(train_data["user_id"])
item_n = np.max([np.max(train_data["item_id"]), np.max(test_data["item_id"])])
knowledge_n = np.max(list(knowledge_set)) + 1


def build_sequential_dataset(df, user_history, item2knowledge, knowledge_n, config, shuffle):
    sample_n = len(df)
    max_seq_len = config.max_seq_len

    user_array = np.empty(sample_n, dtype=np.int64)
    seq_item_array = np.zeros((sample_n, max_seq_len), dtype=np.int64)
    seq_score_array = np.full((sample_n, max_seq_len), -1.0, dtype=np.float32)
    seq_time_array = np.zeros((sample_n, max_seq_len), dtype=np.float32)
    target_item_array = np.empty(sample_n, dtype=np.int64)
    target_score_array = np.empty(sample_n, dtype=np.float32)
    target_time_array = np.empty(sample_n, dtype=np.float32)
    knowledge_emb_array = np.zeros((sample_n, knowledge_n), dtype=np.float32)

    for idx, row in enumerate(df.itertuples(index=False)):
        u = int(row.user_id)
        i = int(row.item_id)
        s = float(row.score)
        t = float(row.timestamp)

        history = user_history[u]
        seq = history[-max_seq_len:]
        hist_len = len(seq)

        if hist_len > 0:
            seq_item_array[idx, :hist_len] = [x[0] for x in seq]
            seq_score_array[idx, :hist_len] = [x[1] for x in seq]
            seq_time_array[idx, :hist_len] = [x[2] for x in seq]

        user_array[idx] = u
        target_item_array[idx] = i
        target_score_array[idx] = s
        target_time_array[idx] = t

        knowledge_codes = item2knowledge.get(i)
        if knowledge_codes is not None and knowledge_codes.size > 0:
            knowledge_emb_array[idx, knowledge_codes] = 1.0

        user_history[u].append((i, s, t))

    dataset = TensorDataset(
        torch.from_numpy(user_array),
        torch.from_numpy(seq_item_array),
        torch.from_numpy(seq_score_array),
        torch.from_numpy(seq_time_array),
        torch.from_numpy(target_item_array),
        torch.from_numpy(knowledge_emb_array),
        torch.from_numpy(target_time_array),
        torch.from_numpy(target_score_array),
    )
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)


split_frames = {
    "train": train_data,
    "test": test_data,
}
future_splits = ["test"]
gauc_user_ids = select_gauc_user_ids(
    split_frames.values(),
    min_interactions_exclusive=getattr(Config, "gauc_min_interactions_exclusive", 100),
)
logging.info(
    "GAUC user filter: interaction_count > %d, selected %d users",
    getattr(Config, "gauc_min_interactions_exclusive", 100),
    len(gauc_user_ids),
)

user_history = defaultdict(list)

logging.info("Building sequential datasets...")
logging.info("Evaluation split build order: %s", " -> ".join(future_splits))
train_loader = build_sequential_dataset(
    split_frames["train"], user_history, item2knowledge, knowledge_n, Config, shuffle=True
)
eval_loaders = {}
for split_name in future_splits:
    eval_loaders[split_name] = build_sequential_dataset(
        split_frames[split_name], user_history, item2knowledge, knowledge_n, Config, shuffle=False
    )

test_loader = eval_loaders["test"]

model_path = Path(getattr(Config, "model_path", "GenCD.snapshot"))
best_model_path = Path(getattr(Config, "best_model_path", "best_GenCD_model.pt"))
model_path.parent.mkdir(parents=True, exist_ok=True)
best_model_path.parent.mkdir(parents=True, exist_ok=True)
cdm = GenCD(Config)
training_summary = cdm.train(train_loader, test_loader, selected_user_ids=gauc_user_ids)
cdm.save(str(model_path))
cdm.load(str(best_model_path))

test_metrics = cdm.eval(test_loader, device=Config.device, selected_user_ids=gauc_user_ids)
test_gauc_text = "nan" if np.isnan(test_metrics["gauc"]) else f"{test_metrics['gauc']:.6f}"
print(
    "test auc: %.6f, gauc: %s, accuracy: %.6f, rmse: %.6f, mae: %.6f"
    % (
        test_metrics["auc"],
        test_gauc_text,
        test_metrics["acc"],
        test_metrics["rmse"],
        test_metrics["mae"],
    )
)
write_result(
    output_file,
    dataset_name,
    split_line,
    "GenCD",
    Config.epochs,
    test_metrics,
    result_hparams(Config),
)

summary_json = getattr(Config, "summary_json", None)
if summary_json:
    run_wall_clock_seconds = time.perf_counter() - run_start
    write_summary(
        Path(summary_json),
        {
            "dataset": dataset_name,
            "dataset_dir": dataset_dir,
            "split_line": split_line,
            "split_spec": split_spec_slug(split_line),
            "model_name": "GenCD",
            "config": {
                "embedding_size": getattr(Config, "embedding_size", None),
                "epochs": getattr(Config, "epochs", None),
                "batch_size": getattr(Config, "batch_size", None),
                "lr": getattr(Config, "lr", None),
                "device": getattr(Config, "device", None),
                "use_time_encoding": getattr(Config, "use_time_encoding", None),
                "dataset_has_timestamp": getattr(Config, "dataset_has_timestamp", None),
                "early_stop_patience": getattr(Config, "early_stop_patience", None),
                "num_layers": getattr(Config, "num_layers", None),
                "num_block": getattr(Config, "num_block", None),
                "lambda_cl": getattr(Config, "lambda_cl", None),
                "lambda_dmr": getattr(Config, "lambda_dmr", None),
                "mask_rate": getattr(Config, "mask_rate", None),
                "max_seq_len": getattr(Config, "max_seq_len", None),
                "dropout_ratio": getattr(Config, "dropout_ratio", None),
            },
            "artifacts": {
                "output_file": output_file,
                "model_path": model_path,
                "best_model_path": best_model_path,
                "checkpoint_path": getattr(Config, "checkpoint_path", None),
            },
            "training": training_summary,
            "test_metrics": test_metrics,
            "run_wall_clock_seconds": float(run_wall_clock_seconds),
        },
    )
