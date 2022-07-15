import torch
import pytorch_lightning as pl

from tqdm import tqdm
#from models.mmdensenet import MMDenseNet
import models.DNN_models.models as models

class MMDenseNetLitModel(pl.LightningModule):
    def __init__(self, loss_fn, optimizer, lr, ckpt_path=None):
        super().__init__()
        if ckpt_path:
            self.net = models.MMDenseNet.build_model(ckpt_path, load_state_dict=True)
        else:
            self.net = models.MMDenseNet.build_from_config("./configs/vocals.yaml") # modify this to use different model
        self.example_input_array = torch.rand(1, 2, 1025, 64) # (B, C, F, T >= 10)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, x):
        out = self.net(x)
        return out

    def configure_optimizers(self):
        optimizer = self.optimizer(self.net.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x_mag, x_phase, y_mag, y_phase = batch
        out = self.net(x_mag)
        #out = x_mag * out # mask
        
        loss = self.loss_fn(out, x_mag)
        self.logger.experiment.add_scalars("loss", {"train": loss}, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        x_mag, x_phase, y_mag, y_phase = batch
        out = self.net(x_mag)
        #out = x_mag * out # mask
        
        loss = self.loss_fn(out, x_mag)
        self.log("val_loss", loss)
        self.logger.experiment.add_scalars("loss", {"validation": loss}, self.global_step)
        return out

    def test_step(self, batch, batch_idx):
        x_mag, x_phase, y_mag, y_phase = batch
        out = self.net(x_mag)
        
        loss = self.loss_fn(out, x_mag)
        self.logger.experiment.add_scalars("loss", {"test": loss}, self.global_step)

    def predict_step(self, batch, batch_idx):
        x_mag, x_phase, y_mag, y_phase = batch
        return x_mag
