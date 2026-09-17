import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from utilities import distance_corr

class MLP(nn.Module):
    def __init__(self, input_size, hidden_layers, use_batchnorm=True, dropout=0.0):
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer size")

        layers = []
        in_features = input_size
        for out_features in hidden_layers:
            layers.append(nn.Linear(in_features, out_features))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = out_features

        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class ABCDLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_layers,
        learning_rate=1e-3,
        bce_weight=1.0,
        disco_lambda=0.0,
        flavor="single",
        use_batchnorm=True,
        dropout=0.0,
        weight_decay=1e-2,
        label_smoothing=0.0,
        use_lr_scheduler=True,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.5,
        lr_scheduler_min_lr=1e-6,
    ):
        super().__init__()
        if flavor not in {"single", "double"}:
            raise ValueError("flavor must be either 'single' or 'double'")

        if flavor == "single":
            self.model = MLP(
                input_size=input_size,
                hidden_layers=hidden_layers,
                use_batchnorm=use_batchnorm,
                dropout=dropout,
            )
        else:
            self.model = nn.ModuleList(
                [
                    MLP(
                        input_size=input_size,
                        hidden_layers=hidden_layers,
                        use_batchnorm=use_batchnorm,
                        dropout=dropout,
                    ),
                    MLP(
                        input_size=input_size,
                        hidden_layers=hidden_layers,
                        use_batchnorm=use_batchnorm,
                        dropout=dropout,
                    ),
                ]
            )

        self.learning_rate = learning_rate
        self.bce_weight = bce_weight
        self.disco_lambda = disco_lambda
        self.flavor = flavor
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_min_lr = lr_scheduler_min_lr

        self.save_hyperparameters()

    def forward(self, x):
        if self.flavor == "single":
            return self.model(x)

        logits_0 = self.model[0](x)
        logits_1 = self.model[1](x)
        return torch.stack([logits_0, logits_1], dim=1)

    def _loss_components(self, batch):
        data, constraint_data, labels, weights = batch
        logits = self(data)

        if logits.ndim == 1:
            logits = logits.unsqueeze(-1)

        smoothed_labels = labels.float()
        if self.label_smoothing > 0:
            smoothed_labels = smoothed_labels * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        scores = torch.sigmoid(logits)

        safe_weights = torch.clamp(weights, min=0.0)
        safe_weights = safe_weights / (torch.mean(safe_weights) + 1e-12)

        bce_components = []
        for head_idx in range(logits.shape[1]):
            bce_components.append(
                F.binary_cross_entropy_with_logits(
                    logits[:, head_idx],
                    smoothed_labels,
                    weight=safe_weights,
                )
            )
        bce = torch.stack(bce_components).sum()

        bkg_mask = labels < 0.5
        disco_term = torch.zeros(1, device=self.device, dtype=logits.dtype).squeeze(0)

        bkg_scores = scores[bkg_mask]
        bkg_constraint = constraint_data[bkg_mask, 0]
        bkg_weights = safe_weights[bkg_mask]

        bkg_score_0 = bkg_scores[:, 0]

        if self.flavor == "single":
            if (torch.max(bkg_score_0) - torch.min(bkg_score_0) > 1e-8) and (
                torch.max(bkg_constraint) - torch.min(bkg_constraint) > 1e-8
            ):
                disco_term = distance_corr(
                    bkg_score_0,
                    bkg_constraint,
                    bkg_weights,
                    power=1,
                )
        else:
            bkg_score_1 = bkg_scores[:, 1]
            if (torch.max(bkg_score_0) - torch.min(bkg_score_0) > 1e-8) and (
                torch.max(bkg_score_1) - torch.min(bkg_score_1) > 1e-8
            ):
                disco_term = distance_corr(bkg_score_0, bkg_score_1, bkg_weights, power=1)

        total_loss = (
            self.bce_weight * bce + self.disco_lambda * disco_term
        )

        return {
            "loss": total_loss,
            "bce": bce,
            "disco": disco_term,
            "bce_components": bce_components,
        }

    def training_step(self, batch, batch_idx):
        losses = self._loss_components(batch)
        self.log("train_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_bce", losses["bce"], on_step=False, on_epoch=True)
        for idx, bce_i in enumerate(losses["bce_components"]):
            self.log(f"train_bce_{idx}", bce_i, on_step=False, on_epoch=True)
        self.log("train_disco", losses["disco"], on_step=False, on_epoch=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses = self._loss_components(batch)
        self.log("val_loss", losses["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_bce", losses["bce"], on_step=False, on_epoch=True)
        for idx, bce_i in enumerate(losses["bce_components"]):
            self.log(f"val_bce_{idx}", bce_i, on_step=False, on_epoch=True)
        self.log("val_disco", losses["disco"], on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        if not self.use_lr_scheduler:
            return optimizer

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            min_lr=self.lr_scheduler_min_lr,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
