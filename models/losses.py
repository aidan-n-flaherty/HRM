from typing import Any, Tuple, Dict, Sequence, Optional
import torch
import torch.nn.functional as F
from torch import nn
import sys
from models.layers import rms_norm


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

        labels = rms_norm(new_carry.current_data["labels"], variance_epsilon=1e-5)
        #labels = new_carry.current_data["labels"]

        preds = rms_norm(outputs["hidden_states"], variance_epsilon=1e-5)

        predict_mask = torch.zeros(preds.shape[1], dtype=torch.bool, device=preds.device)
        for idx in range(int(predict_mask.shape[0]/4), int(predict_mask.shape[0]*3/4)):
            predict_mask[idx] = True

        print(flush=True, file=sys.stderr)

        #for idx in range(0, 4):
        #    print(idx, "label norm:", torch.linalg.norm(labels[:, int(idx * predict_mask.shape[0]/4):int((idx + 1) * predict_mask.shape[0]/4)]), flush=True, file=sys.stderr)

        l = []
        p = []
        for idx in range(0, 4):
            #print(idx, "norm:", torch.linalg.norm(preds[:, int(idx * predict_mask.shape[0]/4):int((idx + 1) * predict_mask.shape[0]/4)]), flush=True, file=sys.stderr)
            
            print("prediction", preds[0, int(idx * predict_mask.shape[0]/4):int((idx + 1) * predict_mask.shape[0]/4)], flush=True, file=sys.stderr)
            print("label", labels[0, int(idx * predict_mask.shape[0]/4):int((idx + 1) * predict_mask.shape[0]/4)], flush=True, file=sys.stderr)

            l.append(labels[:, int(idx * predict_mask.shape[0]/4):int((idx + 1) * predict_mask.shape[0]/4)])
            p.append(preds[:, int(idx * predict_mask.shape[0]/4):int((idx + 1) * predict_mask.shape[0]/4)])
        
        print("d(pred1 - start):", torch.linalg.norm(p[1] - l[0]), "d(pred1 - target):", torch.linalg.norm(p[1] - l[1]), flush=True, file=sys.stderr)
        print("d(pred2 - end):", torch.linalg.norm(p[2] - l[3]), "d(pred2 - target):", torch.linalg.norm(p[2] - l[2]), flush=True, file=sys.stderr)
        print("d(pred0 - start):", torch.linalg.norm(p[0] - l[0]), "d(pred3 - end):", torch.linalg.norm(p[3] - l[3]), flush=True, file=sys.stderr)
        print("d(pred1 - end):", torch.linalg.norm(p[1] - l[3]), "d(pred2 - start):", torch.linalg.norm(p[2] - l[0]), flush=True, file=sys.stderr)
        print("d(start - target1):", torch.linalg.norm(l[1] - l[0]), "d(end - target2):", torch.linalg.norm(l[3] - l[2]), flush=True, file=sys.stderr)

        diff = (preds[:, predict_mask] - labels[:, predict_mask].to(preds.dtype)) ** 2
        per_seq_mse = diff.mean(dim=(-1, -2))

        with torch.no_grad():
            threshold = torch.quantile(per_seq_mse.float(), 0.25)
            seq_is_correct = per_seq_mse <= threshold
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