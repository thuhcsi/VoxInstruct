import os
import torch
import torchaudio
import random
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from encodec import EncodecModel


from model.ar import VoxInstructAR
from model.nar import VoxInstructNAR
from utils.utils import get_config_from_file, sequence_mask, to_device, save_wav, top_k_top_p_filtering, convert_audio
from utils.extract_hubert import HubertWithKmeans
from transformers import AutoTokenizer
from vocos import Vocos


class InferVoxDataset(Dataset):
    def __init__(self, filepath, hp=None, device=None):
        self.filepath = filepath
        self.hp = hp
        self.max_len = hp.max_len
        self.max_text_len = hp.max_text_len
        self.lang_num = hp.lang_num
        self.at_res_num = hp.at_res_num 
        self.bos_id = hp.bos_id
        self.eos_id = hp.eos_id
        self.without_st = hp.without_st
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(hp.mt5_path)
        
        self.hubert = HubertWithKmeans(
            checkpoint_path=f'{hp.hubert_path}/hubert_base_ls960.pt',
            kmeans_path=f'{hp.hubert_path}/hubert_base_ls960_L9_km500.bin',
            target_sample_hz=16000,
            seq_len_multiple_of=320
        )
        self.hubert.eval()
        self.hubert.to(device)
        self.encodec = EncodecModel.encodec_model_24khz()
        self.encodec.set_target_bandwidth(6.0)
        self.encodec.to(device)

        self.samples = self.get_samples(filepath)
        
    def get_samples(self, filepath):
        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f.readlines():
                items = line.strip().split("|")      
                basename, lang_id, instruction, audio_prompt_path = items[0], items[1], items[2], items[3]    
                samples.append(
                    {
                        "basename": basename,
                        "lang_id": lang_id,
                        "instruction": instruction,
                        "audio_prompt_path": audio_prompt_path
                    }
                )   
        
        print(f"loading from {filepath}: {len(samples)} samples")
        
        return samples 

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        basename = self.samples[idx]["basename"]
        text = self.samples[idx]['instruction'].strip().capitalize()
        print(basename, text)

        lang_id = int(self.samples[idx]['lang_id'])
        text_id = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()  # Batch size 1
        if text_id.shape[0] >= self.max_text_len:
            text_id = text_id[:self.max_text_len]
            # <eos> for mt5
            text_id[self.max_text_len-1] = 1
            print("[Note]: instruction text should be less than max_text_len.")

        if os.path.exists(self.samples[idx]['audio_prompt_path']):
            audio_prompt, sr = torchaudio.load(self.samples[idx]['audio_prompt_path'])
            audio_prompt_16khz = convert_audio(audio_prompt, sr, 16000, 1).to(self.device)
            cond_semantic = self.hubert(audio_prompt_16khz)
            cond_semantic = torch.unique_consecutive(cond_semantic).cpu().numpy()
            
            # a trick, to avoid early stop of st prediction
            cond_semantic = cond_semantic[:min(int(cond_semantic.shape[-1]*0.8), cond_semantic.shape[-1]-20)]

            audio_prompt_24khz = convert_audio(audio_prompt, sr, 24000, 1).unsqueeze(0).to(self.device)     
            encoded_frames = self.encodec.encode(audio_prompt_24khz)
            cond_acoustic = encoded_frames[0][0][0]
            cond_acoustic = cond_acoustic.cpu().numpy().squeeze()

        else:
            cond_semantic = None
            cond_acoustic = None

        return basename, text_id, lang_id, cond_semantic, cond_acoustic
        

    # [feature] just support batch_size == 1
    def collate_fn(self, batches):
        temp = []
        
        for basename, text_id, lang_id, cond_semantic, cond_acoustic in batches: 
            # offset
            lang_id = lang_id + 1

            if cond_acoustic is None:
                cond_acoustic = np.zeros(shape=[self.hp.at_res_num, 0], dtype=np.int32)
            else:
                cond_acoustic = cond_acoustic + 1 + self.hp.st_token_num + self.lang_num

            if cond_semantic is not None and not self.without_st:
                cond_semantic = cond_semantic + 1 + self.lang_num       
                seq = np.asarray([self.bos_id] + [lang_id] + list(cond_semantic))
                # full codebook
                full_cond_semantic = np.stack([cond_semantic] * self.at_res_num, axis=1) 
                bos = np.stack([np.asarray([self.bos_id,])] * self.at_res_num, axis=1) 
                lang = np.stack([np.asarray([lang_id,])] * self.at_res_num, axis=1) 
                full_seq = np.concatenate([bos, lang, full_cond_semantic], axis=0) 
            else: 
                seq = np.asarray([self.bos_id] + [lang_id])
                bos = np.stack([np.asarray([self.bos_id,])] * self.at_res_num, axis=1) 
                lang = np.stack([np.asarray([lang_id,])] * self.at_res_num, axis=1) 
                full_seq = np.concatenate([bos, lang], axis=0) 

            segment_id = np.asarray([1] * int(full_seq.shape[0]))

            temp.append([basename, text_id, seq, full_seq, segment_id, cond_acoustic])

        # length padding
        basenames = []
        seqs = []
        full_seqs = []
        seq_lens = []
        segment_ids = []
        text_ids = []
        text_id_lens = []
        cond_acoustics = []
            
        for basename, text_id, seq, full_seq, segment_id, cond_acoustic in temp:
            seq_lens.append(seq.shape[0])
            text_id_lens.append(text_id.shape[0])
            text_id = np.pad(text_id, (0, self.max_text_len - text_id.shape[0]), mode='constant', constant_values=0)

            basenames.append(basename)
            seqs.append(seq)
            full_seqs.append(full_seq)
            segment_ids.append(segment_id)
            text_ids.append(text_id)
            cond_acoustics.append(cond_acoustic)


        # to numpy, to torch
        seqs = torch.from_numpy(np.asarray(seqs))
        seq_lens = torch.from_numpy(np.asarray(seq_lens))
        full_seqs = torch.from_numpy(np.asarray(full_seqs))
        segment_ids = torch.from_numpy(np.asarray(segment_ids))
        text_ids = torch.from_numpy(np.asarray(text_ids))
        text_id_lens = torch.from_numpy(np.asarray(text_id_lens))
        cond_acoustics = torch.from_numpy(np.asarray(cond_acoustics))   
        
        return text_ids, text_id_lens, seqs, seq_lens, full_seqs, segment_ids, cond_acoustics, basenames


@torch.no_grad()
def main(args, device):
    hp = get_config_from_file(args.ar_config).hparams
    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    # prepare models
    ar_model, nar_model, encodec, vocos = prepare_models(args, device)

    #### prepare dataset and inference !!!! 
    testset = InferVoxDataset(args.synth_file, hp=hp, device=device)
    test_loader = DataLoader(testset,
                              num_workers=0,
                              shuffle=False,
                              sampler=None,
                              batch_size=1,
                              collate_fn=testset.collate_fn,
                              )
    
    for i, loaded_data in enumerate(test_loader):
        text_ids, text_id_lens, seqs, seq_lens, full_seqs, segment_ids, cond_acoustics, basenames = to_device(loaded_data, device=device)
        text_attn_mask = sequence_mask(text_id_lens, max_len=hp.max_text_len, device=device)
        text_free_mask = torch.zeros_like(text_attn_mask, device=device)
        at_prompt_len = cond_acoustics.shape[-1]
        
        ##### AR inference
        pred_st_flag = True
        past_key_values_base = None
        past_key_values_st_cfg_text = None
        past_key_values_at_cfg_text = None
        past_key_values_at_cfg_st = None
        
        text_encode = None
        free_text_encode = torch.zeros([1, hp.max_text_len, hp.hidden_dim], device=device).to(torch.bfloat16)
        
        for j in tqdm(range(hp.max_len)):
            # base 
            ar_outputs_base, text_encode = ar_model.predict(
                input_ids=seqs,
                segment_ids=segment_ids,
                text_ids=text_ids,   
                text_attn_mask=text_attn_mask,
                past_key_values=past_key_values_base,
                text_encode=text_encode,
            )           
            cond_logits = ar_outputs_base['logits']
            past_key_values_base = ar_outputs_base['past_key_values']
            
            # 分为预测st序列还是at序列
            if pred_st_flag:
                if args.cfg_st_on_text != 1.0:
                    ar_outputs_st_cfg_text, _ = ar_model.predict(
                        input_ids=seqs,
                        segment_ids=segment_ids,
                        text_ids=text_ids,   
                        text_attn_mask=text_attn_mask,
                        past_key_values=past_key_values_st_cfg_text,
                        text_encode=free_text_encode
                    )
                    uncond_logits = ar_outputs_st_cfg_text['logits']
                    past_key_values_st_cfg_text = ar_outputs_st_cfg_text['past_key_values']
                    logits = uncond_logits + (cond_logits - uncond_logits) * args.cfg_st_on_text
                else: 
                    logits = cond_logits

                logits[:, :, hp.bos_id] = -1e5
                logits[:, :, hp.st_token_num + 1:hp.eos_id] = -1e5
                logits[:, :, hp.eos_id + 1] = -1e5
                
                filtered_logits = top_k_top_p_filtering(logits[0, -1, :], top_k=5, top_p=0.95, temperature=0.8)
                
            else:
                if args.cfg_at_on_text != 1.0:
                    # description prompt全部mask
                    ar_outputs_at_cfg_text, _ = ar_model.predict(
                        input_ids=seqs,
                        segment_ids=segment_ids,
                        text_ids=text_ids,   
                        text_attn_mask=text_attn_mask,
                        past_key_values=past_key_values_at_cfg_text,
                        text_encode=free_text_encode,
                    ) 
                    uncond_logits_text = ar_outputs_at_cfg_text['logits']
                    past_key_values_at_cfg_text = ar_outputs_at_cfg_text['past_key_values']
                    cond_logits = uncond_logits_text + (cond_logits - uncond_logits_text) * (args.cfg_at_on_text)
                
                if args.cfg_at_on_st != 1.0:
                    # st 全部mask
                    ar_outputs_at_cfg_st, _ = ar_model.predict(
                        input_ids=seqs,
                        segment_ids=segment_ids,
                        text_ids=text_ids,   
                        text_attn_mask=text_attn_mask,
                        past_key_values=past_key_values_at_cfg_st,
                        text_encode=text_encode,
                        mask_st=True,
                    )   
                    uncond_logits_st = ar_outputs_at_cfg_st['logits']
                    past_key_values_at_cfg_st = ar_outputs_at_cfg_st['past_key_values']
                    logits = uncond_logits_st + (cond_logits - uncond_logits_st) * (args.cfg_at_on_st)           
                else: 
                    logits = cond_logits

                logits[:, :, hp.bos_id] = -1e5
                logits[:, :, 0:hp.st_token_num + 1] = -1e5
                logits[:, :, hp.eos_id] = -1e5
                
                filtered_logits = top_k_top_p_filtering(logits[0, -1, :], top_k=50, top_p=0.95, temperature=0.8)

            probs = filtered_logits.softmax(dim=-1)
            samples = torch.multinomial(probs, 1).unsqueeze(1).to(device)
                
            # for next infer
            seqs = torch.cat([seqs, samples], dim=1) # [b, t]
    

            if pred_st_flag:
                segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 1], dim=1)
            else:
                segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 2], dim=1)

            # NOTE: indicated that switching from ST to AT
            if samples.item() == hp.eos_id:
                pred_st_flag = False
                print("End predicting semantic tokens at:", j)
                # 拼上cond acoustics
                seqs = torch.cat([seqs, cond_acoustics[:, 0]], dim=-1)
                segment_ids = torch.cat([segment_ids, torch.zeros((1, at_prompt_len), device=device).long() + 2], dim=1)
                st_len = (segment_ids == 1).sum(dim=1)
                past_key_values_base = None


            if samples.item() == hp.eos_id + 1:
                break
            elif j == hp.max_len - 1:
                # too long break
                samples[:, :] = hp.eos_id + 1
                seqs = torch.cat([seqs, samples], dim=1) # [b, t]
                segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 2], dim=1)
                break
        
        ##### NAR inference
        b, t = seqs.shape
        full_seqs = torch.stack([seqs] * hp.at_res_num, dim=1) # [b, n_codebook, t]
        full_seqs[:, :, st_len:st_len+at_prompt_len] = cond_acoustics
        layer_index = torch.ones(size=[b,], device=device) # [b,]
        mask1 = layer_index.unsqueeze(1) > torch.arange(hp.at_res_num, device=device).unsqueeze(0) # [b, 1] > [1, n_codebook] = [b, n_codebook]
        mask2 = (st_len + at_prompt_len).unsqueeze(1) > torch.arange(t, device=device).unsqueeze(0) # [b, 1] > [1, t] = [b, t]
        
        mask = mask1.unsqueeze(2) + mask2.unsqueeze(1) # [b, n_codebbok, 1] + [b, 1, t]
        full_seqs = torch.where(mask, full_seqs, torch.zeros_like(full_seqs))
        for layer_index in range(1, hp.at_res_num):
            full_seqs = nar_model.predict(full_seqs, 
                                       at_prompt_len, 
                                       segment_ids, 
                                       text_ids=text_ids,   
                                       text_attn_mask=text_attn_mask,
                                       layer_index=layer_index, 
                                       iter_step=args.nar_iter_steps,
                                    )
        
        full_seq = full_seqs[:, :, st_len+at_prompt_len:] # [b, n_codbook, t]
        full_seq = (full_seq - hp.st_token_num - hp.lang_num - 1)[:, :, :-1].clamp(0, 1023) # remove pad + text, and the last eos

        basenames[0] = basenames[0].replace('/', '_')
        if args.vocoder == "vocos":
            features = vocos.codes_to_features(full_seq[0])
            bandwidth_id = torch.tensor([2], device=device)  # 6 kbps
            wav = vocos.decode(features, bandwidth_id=bandwidth_id)
            wav = wav.cpu().squeeze(1).squeeze(0).numpy()
            save_wav(wav, os.path.join(args.out_dir, '{}.wav'.format(basenames[0])), sr=24000)
        else:
            wav = encodec.decode([(full_seq, None)]) 
            wav = wav.cpu().squeeze(1).squeeze(0).numpy()
            save_wav(wav, os.path.join(args.out_dir, 'infer-encodec-{}.wav'.format(basenames[0])), sr=24000)


def prepare_models(args, device):
    hp = get_config_from_file(args.ar_config).hparams
    ar_model = VoxInstructAR(hp=hp).to(device)
    ckpt = torch.load(args.ar_ckpt, map_location=device)
    ar_model.load_state_dict(ckpt['model'], strict=True)
    ar_model.to(torch.bfloat16).eval()

    hp = get_config_from_file(args.nar_config).hparams
    nar_model = VoxInstructNAR(hp=hp).to(device)
    ckpt = torch.load(args.nar_ckpt, map_location=device)
    nar_model.load_state_dict(ckpt['model'], strict=True)
    nar_model.to(torch.bfloat16).eval()

    vocos = Vocos.from_hparams(f"{hp.vocos_path}/config.yaml")
    state_dict = torch.load(f"{hp.vocos_path}/pytorch_model.bin", map_location="cpu")  
    encodec_parameters = {
        "feature_extractor.encodec." + key: value
        for key, value in vocos.feature_extractor.encodec.state_dict().items()
    }
    state_dict.update(encodec_parameters)
    vocos.load_state_dict(state_dict)
    vocos.to(device).eval()

    encodec = EncodecModel.encodec_model_24khz(True, Path(f'{hp.encodec_path}')).to(device)
    encodec.overlap = 0
    encodec.set_target_bandwidth(bandwidth=6.0) 

    return ar_model, nar_model, encodec, vocos




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--ar_config", 
                        type=str, 
                        default="configs/train_ar.yaml",
                        help="AR config path")
    parser.add_argument("--nar_config", 
                        type=str, 
                        default="configs/train_nar.yaml",
                        help="NAR config path")
    parser.add_argument('--ar_ckpt',
                        type=str,
                        required=True,
                        help='model ckpt path')
    parser.add_argument('--nar_ckpt',
                        type=str,
                        required=True,
                        help='model ckpt path')
    parser.add_argument('--synth_file',
                        type=str,
                        required=True,
                        help='the file path of synthesized samples')
    parser.add_argument('--out_dir',
                        type=str,
                        default="out_wavs",
                        help='dir of output wav file')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='Inference device, \"cpu\" or \"cuda\"')
    parser.add_argument('--vocoder',
                        type=str,
                        default='vocos',
                        help='Inference vocoder, \"vocos\" or \"encodec\"')  
    parser.add_argument('--cfg_st_on_text',
                        type=float,
                        default=1.0,
                        help='classifier-free guidance factor, default 1.0 does not use cfg (will speed up)')    
    parser.add_argument('--cfg_at_on_text',
                        type=float,
                        default=1.0,
                        help='classifier-free guidance factor, default 1.0 does not use cfg (will speed up)') 
    parser.add_argument('--cfg_at_on_st',
                        type=float,
                        default=1.0,
                        help='classifier-free guidance factor, default 1.0 does not use cfg (will speed up)') 
    parser.add_argument('--nar_iter_steps',
                        type=int,
                        default=1,
                        help='iterative decoding steps in NAR stage')                
    args = parser.parse_args()

    assert 'cpu' in args.device or 'cuda' in args.device, "device must be \"cpu\" or \"cuda\""
    assert 'vocos' in args.vocoder or 'encodec' in args.vocoder, "vocoder must be \"vocos\" or \"encodec\""

    main(args, device=args.device)




