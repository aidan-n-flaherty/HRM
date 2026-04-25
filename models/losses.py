from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn


IGNORE_LABEL_ID = -100
PREDICT_TOKEN_IDS = [1, 2]


class ContinuousACTLossHead(nn.Module):
    def __init__(self, model: nn.Module, dim: int, hidden_size: int):
        super().__init__()
        self.model = model

    def initial_carry(self, batch, **kwargs):
        return self.model.initial_carry(batch, **kwargs)  # type: ignore

    def forward(
        self,
        return_keys: Sequence[str],
        **model_kwargs,
    ) -> Tuple[Any, torch.Tensor, Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], torch.Tensor]:
        new_carry, outputs = self.model(**model_kwargs)

        labels = new_carry.current_data["labels"]

        preds = outputs["hidden_states"]

        predict_mask = torch.zeros(preds.shape[1], dtype=torch.bool, device=preds.device)
        for idx in range(int(predict_mask.shape[0]/4), int(predict_mask.shape[0]*3/4)):
            predict_mask[idx] = True

        diff = (preds[:, predict_mask] - labels[:, predict_mask].to(preds.dtype)) ** 2
        per_seq_mse = diff.mean(dim=(-1, -2))

        with torch.no_grad():
            seq_is_correct = per_seq_mse < 1e-3
            valid_metrics = new_carry.halted

        metrics = {
            "count":           valid_metrics.sum(),
            "mse":             torch.where(valid_metrics, per_seq_mse, 0).sum().detach(),
            "q_halt_accuracy": (valid_metrics & ((outputs["q_halt_logits"] >= 0) == seq_is_correct)).sum(),
            "steps":           torch.where(valid_metrics, new_carry.steps.to(torch.float32), 0).sum(),
        }

        mse_loss = torch.where(valid_metrics, per_seq_mse, 0).sum()

        q_halt_loss = F.binary_cross_entropy_with_logits(
            outputs["q_halt_logits"],
            seq_is_correct.to(outputs["q_halt_logits"].dtype),
            reduction="sum"
        )

        metrics.update({
            "mse_loss":    mse_loss.detach(),
            "q_halt_loss": q_halt_loss.detach(),
        })

        q_continue_loss = 0
        if "target_q_continue" in outputs:
            q_continue_loss = F.binary_cross_entropy_with_logits(
                outputs["q_continue_logits"],
                outputs["target_q_continue"],
                reduction="sum"
            )
            metrics["q_continue_loss"] = q_continue_loss.detach()

        all_outputs = {**outputs, "preds": preds}
        detached_outputs = {k: all_outputs[k].detach() for k in return_keys if k in all_outputs}

        return new_carry, mse_loss + 0.5 * (q_halt_loss + q_continue_loss), metrics, detached_outputs, new_carry.halted.all()