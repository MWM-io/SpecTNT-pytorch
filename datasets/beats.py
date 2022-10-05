import torch as th
import torch.utils.data as tud
import pytorch_lightning as pl


class DummyBeatDataset(tud.Dataset):

    def __init__(self, sample_rate, input_length, hop_length, time_shrinking, mode):
        self.sample_rate = sample_rate
        self.input_length = input_length

        self.target_fps = sample_rate / (hop_length * time_shrinking)
        self.target_nframes = int(input_length * self.target_fps)

        assert mode in ["train", "validation", "test"]
        self.mode = mode

    def __len__(self):
        if self.mode == "train":
            return 80
        elif self.mode == "validation":
            return 10
        elif self.mode == "test":
            return 10

    def __getitem__(self, i):
        if self.mode == "train":
            return {
                'audio': th.zeros(self.input_length * self.sample_rate),
                'targets': th.zeros(self.target_nframes, 3)
            }
        elif self.mode in ["validation", "test"]:
            return {
                'audio': th.zeros(10 * self.input_length * self.sample_rate),
                'targets': th.zeros(10 * self.target_nframes, 3),
                'beats': th.arange(0, 50, 0.5),
                'downbeats': th.arange(0, 50, 2.)
            }


class DummyBeatDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, n_workers, pin_memory, sample_rate, input_length, hop_length, time_shrinking):
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.pin_memory = pin_memory
        self.sample_rate = sample_rate
        self.input_length = input_length
        self.hop_length = hop_length
        self.time_shrinking = time_shrinking

    def setup(self, stage):
        self.train_set = DummyBeatDataset(
            self.sample_rate,
            self.input_length,
            self.hop_length,
            self.time_shrinking,
            "train"
        )
        self.val_set = DummyBeatDataset(
            self.sample_rate,
            self.input_length,
            self.hop_length,
            self.time_shrinking,
            "validation"
        )
        self.test_set = DummyBeatDataset(
            self.sample_rate,
            self.input_length,
            self.hop_length,
            self.time_shrinking,
            "test"
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
