from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from  pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from utils.dataset import MusDB18Dataset
from utils.MMDenseNetLitModel import MMDenseNetLitModel
from utils.data_utils import FeatureTransform, InverseFeatureTransform
import musdb

def main(args):
    # data
    feat_transform = FeatureTransform(args.n_fft, args.hop_len)
    ifeat_transform = InverseFeatureTransform(args.n_fft, args.hop_len)

    # trainer
    loss_fn = F.mse_loss
    optimizer = torch.optim.RMSprop
    callbacks = [
            EarlyStopping(monitor="val_loss", patience=50),
            ModelCheckpoint(monitor="val_loss", save_last=True),
            ]

    model = MMDenseNetLitModel(loss_fn, optimizer, args.lr, args.ckpt)
    #model.to_onnx("model.onnx", export_params=True, opset_version=11)

    if args.mode == "predict":
        args.output.mkdir(parents=True, exist_ok=True)
        #model.load_from_checkpoint(args.ckpt)
        if args.input is not None: # audio file is given
            wav, sr = torchaudio.load(args.input)
            if args.mono:
                wav = wav.mean(dim=0, keepdim=True)
            with torch.no_grad():
                mag, phase = feat_transform(wav)
                x = mag.unsqueeze(0)
                preds = []
                for x_start in tqdm(range(0, x.shape[-1], args.num_frames)):
                    x_in = x[:, :, :, x_start: min(x_start + args.num_frames, x.shape[-1])]
                    pred = model(x_in)
                    #pred *= x_in # mask prediction
                    preds.append(pred)
                pred_mag = torch.cat(preds, dim=-1).squeeze(0)
                pred_wav = ifeat_transform(pred_mag, phase)
            torchaudio.save(args.output / args.input.name, pred_wav, sr, encoding="PCM_S", bits_per_sample=16)
        else: # If audio file is not given, use MusDB18 test set
            mus = musdb.DB(root=args.musdb, subsets="test")
            for track in tqdm(mus):
                with torch.no_grad():
                    mag, phase = feat_transform(torch.from_numpy(track.audio.T).float())
                    pred_mag = model(mag.unsqueeze(dim=0)).squeeze(dim=0)
                    pred_wav = ifeat_transform(pred_mag, phase)
                    (args.output / "test" / track.name).mkdir(parents=True, exist_ok=True)
                    torchaudio.save((args.output / "test" / track.name / args.target).with_suffix(".wav"), pred_wav, args.sr, 
                            format="wav", encoding="PCM_S", bits_per_sample=16)
    else:
        dataset = {}
        dataloader = {}
        trainer = pl.Trainer(
                devices=args.num_devices, 
                accelerator="gpu", 
                default_root_dir=args.ckpt,
                #auto_lr_find=True,
                #auto_scale_batch_size=True,
                enable_progress_bar=True,
                log_every_n_steps=1,
                max_epochs=200,
                #precision=16,
                callbacks=callbacks,
                )

        if args.mode == "train":
            # prepare data
            dataset["train"] = MusDB18Dataset(args.musdb, "train", args.target, args.num_frames, args.mono, feat_transform)
            dataset["valid"] = MusDB18Dataset(args.musdb, "valid", args.target, args.num_frames, args.mono, feat_transform)
            dataloader["train"] = DataLoader(dataset["train"], 
                    batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=args.num_workers)
            dataloader["valid"] = DataLoader(dataset["valid"], 
                    batch_size=args.batch_size, pin_memory=True, num_workers=args.num_workers)

            # train
            trainer.fit(model, dataloader["train"], dataloader["valid"])

        if args.mode == "test":
            # prepare data
            dataset["test"] = MusDB18Dataset(args.musdb, "test", args.target, args.num_frames, args.mono, feat_transform)
            dataloader["test"] = DataLoader(dataset["test"], batch_size=1, pin_memory=True, num_workers=args.num_workers)

            # test
            trainer.test(model, dataloader["test"])

def parse_args():
    parser = ArgumentParser(description="Please choose one of the mode from [train, test, predict]. In predict mode, MusDB18 test set would be used if no audio file given.")

    # Setting
    parser.add_argument("mode", type=str, choices=["train", "test", "predict"], help="to train, test, or predict")
    parser.add_argument("-i", "--input", type=Path, help="path to the audio file to separate")
    parser.add_argument("-o", "--output", type=Path, default="./output", help="path to store the separated audio file")

    # Data
    parser.add_argument("--musdb", type=Path, default="/mnt/HDD1/hansli112/data/musdb18", 
            help="path to the MUSDB18 dataset root folder")
    parser.add_argument("-t", "--target", type=str, default="accompaniment", 
            help="the source to separate (vocals, bass, drums, other, accompaniment)") # ["vocals", "bass", "drums", "other", "accompaniment"]
    parser.add_argument("--sr", type=int, default=44100, help="sampling rate")
    parser.add_argument("--mono", type=bool, default=False, help="whether to convert audio to mono")
    parser.add_argument("--num_frames", type=int, default=256, help="the number of frames as the input of the model")
    parser.add_argument("--num_workers", type=int, default=8, help="how many subprocesses to use for data loading")

    # Fourier Transform
    parser.add_argument("--n_fft", type=int, default=2048, help="the size of Fourier transform")
    parser.add_argument("--hop_len", type=int, default=1024, help="the distance between neighboring sliding window frames")

    # Training
    parser.add_argument("--ckpt", type=Path, default="/mnt/hdd/hansli112/MMDenseNet", help="path to store/load checkpoint")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num_devices", type=int, default=2, help="number of devices")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
