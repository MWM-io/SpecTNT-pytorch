import pytorch_lightning as pl


class BaseModel(pl.LightningModule):
    def __init__(self, feature_extractor, net, optimizer, lr_scheduler, criterion, datamodule):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.net = net
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = criterion
        self.datamodule = datamodule

    def configure_optimizers(self):
        if self.lr_scheduler is None:
            return {"optimizer": self.optimizer}
        else:
            return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "val_loss"}

    @staticmethod
    def _classname(obj, lower=True):
        if hasattr(obj, '__name__'):
            name = obj.__name__
        else:
            name = obj.__class__.__name__
        return name.lower() if lower else name
