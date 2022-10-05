import torch as th
import pytorch_lightning as pl
import torch.utils.data as tud


class DummyTaggingDataset(tud.Dataset):
    def __init__(
        self,
        sample_rate,
        input_length,
        mode
    ):
        self.num_samples = int(input_length * sample_rate)
        self.dummy_tags = th.zeros(50)
        self.dummy_tags[0] = 1
        
        assert mode in ["train", "validation", "test"]
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return 80
        elif self.mode == "validation":
            return 10
        elif self.mode == "test":
            return 10

    def __getitem__(self, index):
        if self.mode == "train":
            return {
                "audio": th.zeros(self.num_samples),
                "targets": self.dummy_tags
            }
        elif self.mode in ["validation", "test"]:
            return {
                "audio": th.zeros(10 * self.num_samples),
                "targets": self.dummy_tags
            }


class DummyTaggingDataModule(pl.LightningDataModule):
    def __init__(
        self,
        sample_rate,
        input_length,
        batch_size,
        n_workers,
        pin_memory
    ):
        self.sample_rate = sample_rate
        self.input_length = input_length
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory

    def setup(self, stage):
        self.train_set = DummyTaggingDataset(
            sample_rate=self.sample_rate,
            input_length=self.input_length,
            mode="train"
        )
        self.val_set = DummyTaggingDataset(
            sample_rate=self.sample_rate,
            input_length=self.input_length,
            mode="validation"
        )
        self.test_set = DummyTaggingDataset(
            sample_rate=self.sample_rate,
            input_length=self.input_length,
            mode="test"
        )

    def train_dataloader(self):
        return tud.DataLoader(self.train_set,
                              batch_size=self.batch_size,
                              pin_memory=self.pin_memory,
                              shuffle=True,
                              num_workers=self.n_workers)

    def val_dataloader(self):
        return tud.DataLoader(self.val_set,
                              batch_size=1,
                              pin_memory=self.pin_memory,
                              shuffle=False,
                              num_workers=self.n_workers)

    def test_dataloader(self):
        return tud.DataLoader(self.test_set,
                              batch_size=1,
                              pin_memory=self.pin_memory,
                              shuffle=False,
                              num_workers=self.n_workers)
