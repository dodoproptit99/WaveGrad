import os
import json
import argparse
import numpy as np

import torch
import torchaudio

from tqdm import tqdm
from datetime import datetime
import IPython.display as ipd
from model import WaveGrad
from benchmark import compute_rtf, iters_schedule_grid_search
from utils import ConfigWrapper, show_message, str2bool, parse_filelist
from data import MelSpectrogramFixed, AudioDataset
from time import time

def get_mel(config, model):
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

    dataset = AudioDataset(config, training=False)
    test_batch = dataset.sample_test_batch(1)

    # n_iter = 25
    # path_to_store_schedule = f'schedules/default/{n_iter}iters.pt'

    # iters_best_schedule, stats = iters_schedule_grid_search(
    #     model, config,
    #     n_iter=n_iter,
    #     betas_range=(1e-06, 0.01),
    #     test_batch_size=1, step=1,
    #     path_to_store_schedule=path_to_store_schedule,
    #     save_stats_for_grid=True,
    #     verbose=True, n_jobs=4
    # )
    
    i=0
    for test in tqdm(test_batch):
        mel = mel_fn(test[None].cuda())
        start = datetime.now()
        t = time()
        outputs = model.forward(mel, store_intermediate_states=False)
        end = datetime.now()
        print("Time infer: ",str(time()-t))
        outputs = outputs.cpu().squeeze()
        save_path = str(i)+'.wav'
        i+=1
        torchaudio.save(
            save_path, outputs, sample_rate=config.data_config.sample_rate
        )
        inference_time = (end - start).total_seconds()
        rtf = compute_rtf(outputs, inference_time, sample_rate=config.data_config.sample_rate)
        show_message(f'Done. RTF estimate:{np.std(rtf)}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', required=True,
        type=str, help='configuration file path'
    )
    parser.add_argument(
        '-ch', '--checkpoint_path',
        required=True, type=str, help='checkpoint path'
    )
    parser.add_argument(
        '-ns', '--noise_schedule_path', required=True, type=str,
        help='noise schedule, should be just a torch.Tensor array of shape [n_iter]'
    )
    # parser.add_argument(
    #     '-m', '--mel_filelist', required=True, type=str,
    #     help='mel spectorgram filelist, files of which should be just a torch.Tensor array of shape [n_mels, T]'
    # )
    parser.add_argument(
        '-v', '--verbose', required=False, type=str2bool,
        nargs='?', const=True, default=True, help='verbosity level'
    )
    args = parser.parse_args()

    # Trying to run inference on GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize config
    with open(args.config) as f:
        config = ConfigWrapper(**json.load(f))

    # Initialize the model
    model = WaveGrad(config)
    model.load_state_dict(torch.load(args.checkpoint_path,map_location=torch.device(device))['model'], strict=False)
    
    # Set noise schedule
    noise_schedule = torch.load(args.noise_schedule_path)
    # print(noise_schedule)
    n_iter = noise_schedule.shape[-1]
    init_fn = lambda **kwargs: noise_schedule
    init_kwargs = {'steps': n_iter}
    model.set_new_noise_schedule(init_fn, init_kwargs)

    # device = 'cpu'
    model = model.to(device)
    get_mel(config, model)

    # # Inference
    # filelist = parse_filelist(args.mel_filelist)
    # rtfs = []
    # for mel_path in (tqdm(filelist, leave=False) if args.verbose else filelist):
    #     with torch.no_grad():
    #         mel = torch.load(mel_path, map_location=torch.device(device)).unsqueeze(0).to(device)
    #         temp = mel[0]
    #         start = datetime.now()
    #         outputs = model.forward(temp, store_intermediate_states=False)
    #         end = datetime.now()

    #         outputs = outputs.cpu().squeeze()
    #         baseidx = os.path.basename(os.path.abspath(mel_path)).split('_')[-1].replace('.pt', '')
    #         save_path = f'{os.path.dirname(os.path.abspath(mel_path))}/predicted_{baseidx}.wav'
    #         torchaudio.save(
    #             save_path, outputs, sample_rate=config.data_config.sample_rate
    #         )

    #         inference_time = (end - start).total_seconds()
    #         rtf = compute_rtf(outputs, inference_time, sample_rate=config.data_config.sample_rate)
    #         rtfs.append(rtf)

    # show_message(f'Done. RTF estimate: {np.mean(rtfs)} ± {np.std(rtfs)}', verbose=args.verbose)
    