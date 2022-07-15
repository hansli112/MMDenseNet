import numpy as np
import torch
import musdb

from torch.utils.data import Dataset

class MusDB18Dataset(Dataset):
    def __init__(self, musdb_dir=None, mode="train", target="accompaniment", num_frames=64, mono=True, feat_transform=None):
        super().__init__()
        if mode == "train":
            self.mus = musdb.DB(root=musdb_dir, subsets="train", split="train")
        elif mode == "valid":
            self.mus = musdb.DB(root=musdb_dir, subsets="train", split="valid")
        else: # mode == test or predict
            self.mus = musdb.DB(root=musdb_dir, subsets="test")
        self.mode = mode
        self.target = target # target -> ["vocals", "bass", "drums", "other", "accompaniment"]
        self.num_frames = num_frames # the number of frames as the input of the model
        self.mono = mono
        self.feat_transform = feat_transform

    def __len__(self):
        return len(self.mus)

    def __getitem__(self, idx):
        track = self.mus[idx]
        x, y = track.audio.T, track.targets[self.target].audio.T # channel x sample
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        
        if self.mono:
            x, y = x.mean(dim=0, keepdim=True), y.mean(dim=0, keepdim=True)
        x_mag, x_phase = self.feat_transform(x) # channel x n_fft x n_frame
        y_mag, y_phase = self.feat_transform(y)

        if self.mode != "predict":
            frame_s = np.random.randint(x_mag.shape[2] - self.num_frames) # the start index of frame
            num_frames = self.num_frames
            x_mag, x_phase = x_mag[:, :, frame_s: frame_s + num_frames], x_phase[:, :, frame_s: frame_s + num_frames]
            y_mag, y_phase = y_mag[:, :, frame_s: frame_s + num_frames], y_phase[:, :, frame_s: frame_s + num_frames]

        return x_mag, x_phase, y_mag, y_phase
