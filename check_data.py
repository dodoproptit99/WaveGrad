import json
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from tqdm import tqdm

from data import MelSpectrogramFixed, AudioDataset
from utils import ConfigWrapper, parse_filelist


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, type=str, help='configuration file')
    parser.add_argument('-f', '--filelist', required=True, type=str, help='audios filelist path')
    args = parser.parse_args()

    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))
    
    mel_fn = MelSpectrogramFixed(
        sample_rate=config.data_config.sample_rate,
        n_fft=config.data_config.n_fft,
        win_length=config.data_config.win_length,
        hop_length=config.data_config.hop_length,
        f_min=config.data_config.f_min,
        f_max=config.data_config.f_max,
        n_mels=config.data_config.n_mels,
        window_fn=torch.hann_window
    ).cuda()

    dataset = AudioDataset(config, training=True)
    dataset.filelist_path = args.filelist
    dataset.audio_paths = parse_filelist(dataset.filelist_path)

    loader = DataLoader(dataset, batch_size=48)

    nans, infs = [], []
    for batch in tqdm(loader, total=int(np.ceil(len(dataset)/48))):
        batch = batch.cuda()
        mels = mel_fn(batch)

        nan_mask = torch.isnan(mels)
        inf_mask = torch.isinf(mels)

        nans.append(nan_mask.sum().cpu())
        infs.append(inf_mask.sum().cpu())
    
    print(f'Dataset has nans: {any([item != 0 for item in nans])}')
    print(f'Dataset has infs: {any([item != 0 for item in infs])}')
