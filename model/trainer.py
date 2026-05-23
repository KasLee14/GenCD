# model/trainer.py
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from EduCDM import CDM
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from tqdm import tqdm

from model.net import Net


def compute_group_auc(y_true, y_pred, user_ids, selected_user_ids=None):
    selected = None if selected_user_ids is None else set(selected_user_ids)
    grouped = {}

    for user_id, truth, pred in zip(user_ids, y_true, y_pred):
        user_id = int(user_id)
        if selected is not None and user_id not in selected:
            continue
        grouped.setdefault(user_id, {"y_true": [], "y_pred": []})
        grouped[user_id]["y_true"].append(float(truth))
        grouped[user_id]["y_pred"].append(float(pred))

    weighted_auc = 0.0
    total_weight = 0
    for group in grouped.values():
        labels = np.asarray(group["y_true"], dtype=np.float32)
        if np.unique(labels).size < 2:
            continue
        preds = np.asarray(group["y_pred"], dtype=np.float32)
        weight = int(labels.size)
        weighted_auc += roc_auc_score(labels, preds) * weight
        total_weight += weight

    if total_weight == 0:
        return float("nan")
    return float(weighted_auc / total_weight)


def compute_metrics(y_true, y_pred, user_ids, selected_user_ids=None):
    return {
        "auc": roc_auc_score(y_true, y_pred),
        "gauc": compute_group_auc(y_true, y_pred, user_ids, selected_user_ids=selected_user_ids),
        "acc": accuracy_score(y_true, np.array(y_pred) >= 0.5),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "mae": mean_absolute_error(y_true, y_pred),
    }


class GenCD(CDM):
    def __init__(self, config):
        super(GenCD, self).__init__()
        self.config = config
        self.ncdm_net = Net(config)

    def _torch_load(self, filepath):
        load_kwargs = {"map_location": self.config.device}
        try:
            return torch.load(filepath, weights_only=True, **load_kwargs)
        except TypeError:
            return torch.load(filepath, **load_kwargs)

    def _move_optimizer_state_to_device(self, optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(self.config.device)

    def _training_checkpoint_path(self):
        checkpoint_path = getattr(self.config, "checkpoint_path", None)
        if not checkpoint_path:
            return None
        return Path(checkpoint_path)

    def _save_training_checkpoint(
        self,
        optimizer,
        next_epoch,
        best_gauc,
        best_epoch,
        no_improve_epochs,
        epoch_records,
        stop_reason,
    ):
        checkpoint_path = self._training_checkpoint_path()
        if checkpoint_path is None:
            return

        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "gencd_training_checkpoint_v1",
                "next_epoch": int(next_epoch),
                "model_state_dict": self.ncdm_net.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_gauc": None if best_gauc is None else float(best_gauc),
                "best_epoch": None if best_epoch is None else int(best_epoch),
                "no_improve_epochs": int(no_improve_epochs),
                "epoch_records": epoch_records,
                "requested_epochs": int(self.config.epochs),
                "stop_reason": stop_reason,
            },
            checkpoint_path,
        )
        logging.info("save training checkpoint to %s" % checkpoint_path)

    def _load_checkpoint_resume_state(self, optimizer):
        if not bool(getattr(self.config, "resume_checkpoint", False)):
            return None

        checkpoint_path = self._training_checkpoint_path()
        if checkpoint_path is None or not checkpoint_path.exists():
            return None

        checkpoint = self._torch_load(checkpoint_path)
        self.ncdm_net.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
            self._move_optimizer_state_to_device(optimizer)

        best_gauc = checkpoint.get("best_gauc")
        if best_gauc is not None:
            best_gauc = float(best_gauc)

        return {
            "source": str(checkpoint_path),
            "start_epoch": int(checkpoint.get("next_epoch", 0)),
            "best_gauc": best_gauc,
            "best_epoch": checkpoint.get("best_epoch"),
            "no_improve_epochs": int(checkpoint.get("no_improve_epochs", 0)),
            "epoch_records": list(checkpoint.get("epoch_records", [])),
        }

    def _load_best_model_resume_state(self):
        if not bool(getattr(self.config, "resume_from_best", False)):
            return None

        best_model_path = Path(getattr(self.config, "best_model_path", "best_GenCD_model.pt"))
        if not best_model_path.exists():
            return None

        self.load(str(best_model_path))
        best_gauc = getattr(self.config, "resume_best_gauc", None)
        if best_gauc is not None:
            best_gauc = float(best_gauc)

        return {
            "source": str(best_model_path),
            "start_epoch": int(getattr(self.config, "resume_start_epoch", 0) or 0),
            "best_gauc": best_gauc,
            "best_epoch": getattr(self.config, "resume_best_epoch", None),
            "no_improve_epochs": int(getattr(self.config, "resume_no_improve_epochs", 0) or 0),
            "epoch_records": [],
        }

    def train(self, train_data, test_data=None, silence=False, selected_user_ids=None):
        print(self.config.device)
        self.ncdm_net = self.ncdm_net.to(self.config.device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=self.config.lr, weight_decay=1e-5)
        run_label = getattr(self.config, "run_label", getattr(self.config, "dataset_name", "dataset"))

        resume_source = None
        start_epoch = 0
        best_gauc = None
        best_epoch = None
        early_stop_patience = int(getattr(self.config, "early_stop_patience", 5))
        no_improve_epochs = 0
        epoch_records = []
        checkpoint_resume_state = self._load_checkpoint_resume_state(optimizer)
        best_model_resume_state = None if checkpoint_resume_state is not None else self._load_best_model_resume_state()
        resume_state = checkpoint_resume_state or best_model_resume_state
        if resume_state is not None:
            resume_source = resume_state["source"]
            start_epoch = min(max(int(resume_state["start_epoch"]), 0), int(self.config.epochs))
            best_gauc = resume_state["best_gauc"]
            best_epoch = resume_state["best_epoch"]
            if best_epoch is not None:
                best_epoch = int(best_epoch)
            no_improve_epochs = int(resume_state["no_improve_epochs"])
            epoch_records = list(resume_state["epoch_records"])
            print(
                f"--> [{run_label}] [Resume] Loaded {resume_source}; "
                f"continuing from epoch {start_epoch}."
            )

        stop_reason = "completed_requested_epochs"
        training_start = time.perf_counter()

        for epoch_i in range(start_epoch, self.config.epochs):
            epoch_start = time.perf_counter()
            self.ncdm_net.train()
            epoch_losses = []
            dmr_losses = []
            gcl_losses = []

            for batch_data in tqdm(train_data, f"{run_label} | Epoch {epoch_i}"):
                (
                    user_id,
                    seq_item_id,
                    seq_score,
                    seq_time,
                    target_item_id,
                    target_knowledge_emb,
                    target_time,
                    y,
                ) = batch_data

                user_id = user_id.to(self.config.device)
                seq_item_id = seq_item_id.to(self.config.device)
                seq_score = seq_score.to(self.config.device)
                seq_time = seq_time.to(self.config.device)
                target_item_id = target_item_id.to(self.config.device)
                target_knowledge_emb = target_knowledge_emb.to(self.config.device)
                target_time = target_time.to(self.config.device)
                y = y.to(self.config.device)

                prob_matrix = torch.rand(seq_item_id.shape, device=self.config.device)
                item_mask = (prob_matrix < self.config.mask_rate) & (seq_item_id != 0)

                pred, dmr_loss, graph_cl_loss = self.ncdm_net(
                    user_id,
                    seq_item_id,
                    seq_score,
                    seq_time,
                    target_item_id,
                    target_knowledge_emb,
                    target_time,
                    item_mask,
                )

                main_loss = loss_function(pred, y)
                total_loss = main_loss + self.config.lambda_dmr * dmr_loss + self.config.lambda_cl * graph_cl_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ncdm_net.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_losses.append(main_loss.item())
                if isinstance(dmr_loss, torch.Tensor):
                    dmr_losses.append(dmr_loss.item())
                if isinstance(graph_cl_loss, torch.Tensor):
                    gcl_losses.append(graph_cl_loss.item())

            train_seconds = time.perf_counter() - epoch_start
            avg_main = float(np.mean(epoch_losses))
            avg_dmr = float(np.mean(dmr_losses)) if dmr_losses else 0.0
            avg_gcl = float(np.mean(gcl_losses)) if gcl_losses else 0.0
            print(
                f"[{run_label}][Epoch {epoch_i} Loss] "
                f"main_loss: {avg_main:.6f}, dmr: {avg_dmr:.6f}, gcl: {avg_gcl:.6f}"
            )

            eval_seconds = 0.0
            metrics = None
            improved = False
            if test_data is not None and (epoch_i + 1) % 1 == 0:
                eval_start = time.perf_counter()
                metrics = self.eval(
                    test_data,
                    device=self.config.device,
                    selected_user_ids=selected_user_ids,
                )
                eval_seconds = time.perf_counter() - eval_start
                gauc_text = "nan" if np.isnan(metrics["gauc"]) else f"{metrics['gauc']:.6f}"
                print(
                    f"[{run_label}][Epoch {epoch_i} Eval] auc: {metrics['auc']:.6f}, "
                    f"gauc: {gauc_text}, acc: {metrics['acc']:.6f}, "
                    f"rmse: {metrics['rmse']:.6f}, mae: {metrics['mae']:.6f}"
                )
                current_gauc = metrics["gauc"]
                if best_gauc is None:
                    improved = True
                elif not np.isnan(current_gauc) and (np.isnan(best_gauc) or current_gauc > best_gauc):
                    improved = True

                if improved:
                    best_gauc = current_gauc
                    best_epoch = epoch_i
                    no_improve_epochs = 0
                    self.save(getattr(self.config, "best_model_path", "best_GenCD_model.pt"))
                    if np.isnan(best_gauc):
                        print(
                            f"--> [{run_label}] [New Best Model Saved] "
                            "GAUC is nan; saved current model as the initial fallback checkpoint."
                        )
                    else:
                        print(f"--> [{run_label}] [New Best Model Saved] GAUC improved to {best_gauc:.6f}")
                else:
                    no_improve_epochs += 1
                    print(
                        f"--> [{run_label}] [Early Stop Counter] no improvement for "
                        f"{no_improve_epochs}/{early_stop_patience} epoch(s)."
                    )
                    if no_improve_epochs >= early_stop_patience:
                        stop_reason = f"early_stop_no_gauc_improvement_{early_stop_patience}"
                        print(
                            f"--> [{run_label}] [Early Stopping] Stop training at epoch {epoch_i} "
                            f"because GAUC did not improve for {early_stop_patience} consecutive epochs."
                        )
                        epoch_seconds = time.perf_counter() - epoch_start
                        print(
                            f"[{run_label}][Epoch {epoch_i} Timing] "
                            f"train_seconds: {train_seconds:.3f}, eval_seconds: {eval_seconds:.3f}, "
                            f"epoch_seconds: {epoch_seconds:.3f}"
                        )
                        epoch_records.append(
                            {
                                "epoch_index": int(epoch_i),
                                "train_seconds": float(train_seconds),
                                "eval_seconds": float(eval_seconds),
                                "epoch_seconds": float(epoch_seconds),
                                "main_loss": float(avg_main),
                                "dmr_loss": float(avg_dmr),
                                "gcl_loss": float(avg_gcl),
                                "eval_metrics": None
                                if metrics is None
                                else {key: float(value) for key, value in metrics.items()},
                                "is_best": bool(improved),
                            }
                        )
                        self._save_training_checkpoint(
                            optimizer,
                            epoch_i + 1,
                            best_gauc,
                            best_epoch,
                            no_improve_epochs,
                            epoch_records,
                            stop_reason,
                        )
                        break

            epoch_seconds = time.perf_counter() - epoch_start
            print(
                f"[{run_label}][Epoch {epoch_i} Timing] "
                f"train_seconds: {train_seconds:.3f}, eval_seconds: {eval_seconds:.3f}, "
                f"epoch_seconds: {epoch_seconds:.3f}"
            )
            epoch_records.append(
                {
                    "epoch_index": int(epoch_i),
                    "train_seconds": float(train_seconds),
                    "eval_seconds": float(eval_seconds),
                    "epoch_seconds": float(epoch_seconds),
                    "main_loss": float(avg_main),
                    "dmr_loss": float(avg_dmr),
                    "gcl_loss": float(avg_gcl),
                    "eval_metrics": None
                    if metrics is None
                    else {key: float(value) for key, value in metrics.items()},
                    "is_best": bool(improved),
                }
            )
            self._save_training_checkpoint(
                optimizer,
                epoch_i + 1,
                best_gauc,
                best_epoch,
                no_improve_epochs,
                epoch_records,
                stop_reason,
            )

            if stop_reason != "completed_requested_epochs":
                break

        total_seconds = time.perf_counter() - training_start
        train_epoch_seconds = [record["train_seconds"] for record in epoch_records]
        completed_epoch_count = max(
            [int(record["epoch_index"]) + 1 for record in epoch_records],
            default=start_epoch,
        )
        return {
            "requested_epochs": int(self.config.epochs),
            "completed_epochs": int(completed_epoch_count),
            "best_epoch": None if best_epoch is None else int(best_epoch),
            "best_gauc": None if best_gauc is None else float(best_gauc),
            "stop_reason": stop_reason,
            "resumed": resume_source is not None,
            "resume_source": resume_source,
            "start_epoch": int(start_epoch),
            "total_training_seconds": float(total_seconds),
            "avg_train_epoch_seconds": float(np.mean(train_epoch_seconds)) if train_epoch_seconds else 0.0,
            "min_train_epoch_seconds": float(np.min(train_epoch_seconds)) if train_epoch_seconds else 0.0,
            "max_train_epoch_seconds": float(np.max(train_epoch_seconds)) if train_epoch_seconds else 0.0,
            "epoch_records": epoch_records,
        }

    def eval(self, test_data, device="cpu", selected_user_ids=None):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred, user_ids = [], [], []
        run_label = getattr(self.config, "run_label", getattr(self.config, "dataset_name", "dataset"))

        with torch.no_grad():
            for batch_data in tqdm(test_data, f"{run_label} | Evaluating"):
                (
                    user_id,
                    seq_item_id,
                    seq_score,
                    seq_time,
                    target_item_id,
                    target_knowledge_emb,
                    target_time,
                    y,
                ) = batch_data
                user_ids.extend(user_id.tolist())

                user_id = user_id.to(device)
                seq_item_id = seq_item_id.to(device)
                seq_score = seq_score.to(device)
                seq_time = seq_time.to(device)
                target_item_id = target_item_id.to(device)
                target_knowledge_emb = target_knowledge_emb.to(device)
                target_time = target_time.to(device)

                pred, _, _ = self.ncdm_net(
                    user_id,
                    seq_item_id,
                    seq_score,
                    seq_time,
                    target_item_id,
                    target_knowledge_emb,
                    target_time,
                    item_mask=None,
                )

                y_pred.extend(pred.cpu().tolist())
                y_true.extend(y.tolist())

        return compute_metrics(
            y_true,
            y_pred,
            user_ids,
            selected_user_ids=selected_user_ids,
        )

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        state_dict = self._torch_load(filepath)
        self.ncdm_net.load_state_dict(state_dict)
        logging.info("load parameters from %s" % filepath)
