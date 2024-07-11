import os
import time
from torch import optim, nn, utils, Tensor
import torch
import lightning as L

class LitEMPSN(L.LightningModule):
    def __init__(self, model, mae, mad, mean, train_samples, test_samples, validation_samples, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.lr = lr
        self.criterion = nn.L1Loss(reduction='sum')
        self.mae = mae
        self.mad = mad
        self.mean = mean
        self.train_samples = train_samples
        self.validation_samples = validation_samples
        self.test_samples = test_samples
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.start_epoch_time = None

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)

        return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
            

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        start_forward_time = time.perf_counter()
        pred = self.model(batch)
        self.log('forward_time', time.perf_counter() - start_forward_time)
        loss = self.criterion(pred, (batch.y - self.mean) / self.mad)

        mae = self.criterion(pred * self.mad + self.mean, batch.y)

        self.log("train_loss", loss, on_step=False, on_epoch=True)
        self.training_step_outputs.append(mae)

        #self.log("train_mae", mae, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch.y.size(0))
        return loss

    def on_train_epoch_start(self):
        self.start_epoch_time = time.perf_counter()

    def on_train_epoch_end(self):
        all_preds = torch.stack(self.training_step_outputs).sum()
        self.log("train_mae", all_preds / self.train_samples)
        self.log("epoch_time", time.perf_counter() - self.start_epoch_time)
        self.training_step_outputs.clear()


    def validation_step(self, batch):
        batch = batch.to(self.device)
        pred = self.model(batch)
        loss = self.criterion(pred, (batch.y - self.mean) / self.mad)

        mae = self.criterion(pred * self.mad + self.mean, batch.y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.validation_step_outputs.append(mae)
        return loss

    def on_validation_epoch_end(self):
        all_preds = torch.stack(self.validation_step_outputs).sum()
        self.log("val_mae", all_preds / self.validation_samples)
        self.validation_step_outputs.clear()

    def test_step(self, batch):
        batch = batch.to(self.device)
        pred = self.model(batch)
        loss = self.criterion(pred, (batch.y - self.mean) / self.mad)
        mae = self.criterion(pred * self.mad + self.mean, batch.y)

        self.log("test_loss", loss)
        self.test_step_outputs.append(mae)
        return loss

    def on_test_epoch_end(self):
        all_preds = torch.stack(self.test_step_outputs).sum()
        self.log("test_mae", all_preds / self.test_samples)
        self.test_step_outputs.clear()
