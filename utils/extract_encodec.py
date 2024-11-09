from pathlib import Path
from encodec import EncodecModel
from encodec.utils import convert_audio
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from time import time

import torchaudio
import torch
from tqdm import tqdm
import json
import torch.nn.functional as F


class EncodecDataset(Dataset):
    def __init__(self, folder, suffix='.wav', sample_rate=24000, channels=1):
        super().__init__()
        self.paths = []
        self.suffix = suffix
        self.sample_rate = sample_rate
        self.channels = channels
        folder = Path(folder)
        self.paths = [*folder.rglob(f'*{self.suffix}')]
        assert len(self.paths) > 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        file = self.paths[item]

        try:
            data, sr = torchaudio.load(file)
        except:
            print(file)
            return None
        
        if data.shape[-1] == 0:
            print(file)
            return None

        data = convert_audio(data, sr, self.sample_rate, self.channels).unsqueeze(0)
        out_file = Path(file).with_suffix('.npy')

        output = (data, out_file)
        return output


def collecte_fn(samples):
    samples = list(filter(lambda sample: sample is not None, samples))
    inputs = []
    out_files = []
    input_lens = [int(sample[0].shape[-1]) for sample in samples]
    max_len = max(input_lens)
    for i, sample in enumerate(samples):
        inputs.append(F.pad(sample[0], (0, max_len - input_lens[i]), mode='constant', value=0.))
        out_files.append(sample[1])
    inputs = torch.cat(inputs, dim=0)
    return (inputs, input_lens, out_files)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", 
                        type=str, 
                        required=True,
                        help="Audiofile folder path to be processed, make sure '*.wav' files exist in it or its subfolders.")
    
    args = parser.parse_args()
    input_folder = Path(args.data)
    output_folder = Path(args.data).parent / 'acoustic'

    suffix = '.wav'
    device = 'cuda'
    sampling_rate = 24000
    channels = 1
    num_workers = 8
    accelerate = Accelerator()

    # Instantiate a pretrained EnCodec model
    model = EncodecModel.encodec_model_24khz()
    # The number of codebooks used will be determined bythe bandwidth selected.
    # E.g. for a bandwidth of 6kbps, `n_q = 8` codebooks are used.
    # Supported bandwidths are 1.5kbps (n_q = 2), 3 kbps (n_q = 4), 6 kbps (n_q = 8) and 12 kbps (n_q =16) and 24kbps (n_q=32).
    # For the 48 kHz model, only 3, 6, 12, and 24 kbps are supported. The number
    # of codebooks for each is half that of the 24 kHz model as the frame rate is twice as much.
    model.set_target_bandwidth(6.0)
    model.to(device)
    
    ds = EncodecDataset(input_folder, suffix, sampling_rate, channels)
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collecte_fn, num_workers=num_workers)
    (model, dl) = accelerate.prepare(model, dl)
    

    for batch in tqdm(dl):
        if batch is None:
            continue
            
        with torch.no_grad():
            inputs, input_lens, out_files = batch

            try:
                inputs = inputs.to(device)
                # Extract discrete codes from EnCodec
                encoded_frames = model.encode(inputs)
                codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # [B, n_q, T]
            except:
                print(inputs.shape)
                continue 
            
            codes = codes.cpu().numpy()
            
            for i, out_file in enumerate(out_files):
                out_file = Path(str(out_file).replace(input_folder, output_folder))
                if not out_file.parent is not None:
                    os.makedirs(out_file.parent, exist_ok=True)
                # np.save(out_file, codes[i])
                np.save(out_file, codes[i, :, :(input_lens[i]+319) // 320])
                


