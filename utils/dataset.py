import os
import json
import re
from tqdm import tqdm
from pathlib import Path
import random
from collections import OrderedDict
from typing import Optional  

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class VoxDataset(Dataset):
    """
    VoxDataset class for handling multiple metadata-json files operations.
    """
    def __init__(self, meta_paths: list, hp: dict = None, is_check: bool = False, eval_mode: bool = False):
        """
        Initialize the VoxDataset class.

        Args:
            meta_paths (list): List of dataset paths.
            lang_list (list): Language mapping list.
            hp (dict, optional): Hyperparameters.
            is_check (bool, optional): Flag to indicate if checking is enabled.
            eval_mode (bool, optional): Flag to indicate if evaluation mode is enabled.
        """

        self.eval_mode = eval_mode
        self.hp = hp
        self.max_len = hp.max_len
        self.max_text_len = hp.max_text_len
        self.lang_num = hp.lang_num 
        self.lang_mapping = hp.lang_mapping
        self.at_res_num = hp.at_res_num 
        self.bos_id = hp.bos_id
        self.eos_id = hp.eos_id
        
        self.meta_paths = meta_paths
        
        self.tokenizer = AutoTokenizer.from_pretrained(hp.mt5_path)
        self.without_st = hp.without_st

        self.semantic_dir_paths, self.acoustic_dir_paths = [], []
        for mp in meta_paths:
            dp = Path(os.path.abspath(os.path.dirname(mp)))
            self.semantic_dir_paths.append(dp / "hubert") 
            self.acoustic_dir_paths.append(dp / "acoustic")

        self.metadata_dict = {}
        self._idx_to_key = []
        self._idx_to_class = []

        for class_idx, mp in enumerate(meta_paths):
            metas, keys, classes = self.get_metadata(mp, class_idx, is_check)
            self.metadata_dict.update(metas)
            self._idx_to_key.extend(keys)
            self._idx_to_class.extend(classes)
    
        assert len(self.metadata_dict) == len(self._idx_to_key), "Metadata dictionary and key list length mismatch"
        print(f"loading All datasets: {len(self.metadata_dict)} samples")
        
    
    def get_metadata(self, path: str, class_idx: int = 0, is_check: bool = False) -> tuple[dict, list, list]:
        """
        Load metadata from a given JSON file.

        Args:
            path (str): The path to the metadata file.
            class_idx (int): The class index to assign to all samples.
            is_check (bool): Flag to indicate if checking None data and filter them out. Defaults to False.

        Returns:
            tuple: A tuple containing the loaded metadata (dict), keys (list), and classes (list).
        """
        with open(path, 'r', encoding='utf-8') as f:
            metas = json.load(f)

        if is_check:
            filtered_metas = {}
            for basename in tqdm(list(metas.keys())):
                st, at = self.get_pair_data(metas, class_idx)
                if st is not None and at is not None:
                    filtered_metas[basename] = metas[basename]
                else:
                    print(f"Find numpy None data: {basename}, st is None: {st is None}, at is None: {at is None}")

            print(f"After filtering None data: {len(filtered_metas)} samples")

            # Save the filtered metadata to a new JSON file
            filtered_path = path.replace('.json', '_filter.json')
            with open(filtered_path, 'w', encoding='utf-8') as f:
                json.dump(filtered_metas, f, ensure_ascii=False, indent=4)

            print("Remember to update the metadata file path in CONFIG before next training.")
            metas = filtered_metas

        # distinguish samples from different datasets    
        new_metas = {f"{class_idx}_{key}": value for key, value in metas.items()}    
        keys = list(new_metas.keys()) 
        classes = list([class_idx] * len(keys))
        print(f"loading from {path}: {len(metas)} samples")

        return new_metas, keys, classes

        
    def get_pair_data(self, basename: str, class_idx: int = 0) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieves the semantic and acoustic data pair for a given basename.

        Parameters:
        - basename (str): The basename of the file.
        - class_idx (int): The index of the class to retrieve the data from <dir_paths>.

        Returns:
        - tuple: A tuple containing the semantic and acoustic data arrays, or None if they cannot be loaded.
        """
        # Remove the redundant suffix if it exists.
        basename = basename.replace(".wav", "")
    
        semantic, acoustic = None, None

        try:
            semantic_path = self.semantic_dir_paths[class_idx] / f'{basename}.npy'
            semantic = np.load(semantic_path)
        except Exception as e:
            print(f"Error loading semantic data for {basename}: {e}")

        try:
            acoustic_path = self.acoustic_dir_paths[class_idx] / f'{basename}.npy'
            acoustic = np.load(acoustic_path)
        except Exception as e:
            print(f"Error loading acoustic data for {basename}: {e}")
            
        return semantic, acoustic
        
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.metadata_dict)


    def __getitem__(self, idx: int) -> tuple:
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the text id, semantic id, acoustic id, lang id, basename.
        """
        basename = self._idx_to_key[idx] 
        
        # because we add a prefix dataset class_idx at the beginning of the key
        semantic_id, acoustic_id = self.get_pair_data(basename.split('_', 1)[1], self._idx_to_class[idx])

        if semantic_id is None or acoustic_id is None:
            raise ValueError(f"Data for {basename} could not be loaded.")

        # ST: [T, ], remove duplicated semantic tokens
        semantic_id = torch.unique_consecutive(torch.from_numpy(semantic_id)).numpy()
        # AT: [N, T] -> [T, N]
        acoustic_id = acoustic_id.transpose(1, 0)
        
        # raw transcript
        transcript = self.metadata_dict[basename]["transcript"].strip().capitalize()
        # lang_id
        lang_id = self.lang_mapping[self.metadata_dict[basename]["language"]]
        
        # Determine instruction text
        if "instruction" in self.metadata_dict[basename] and len(self.metadata_dict[basename]["instruction"]) > 0:
            ins_text = random.choice(self.metadata_dict[basename]["instruction"]).strip().capitalize()
        else:
            ins_text = '"{}"'.format(transcript) if random.random() > 0.5 else '“{}”'.format(transcript)

        # During training, randomly choose between the raw transcript and the description by a certain description_free ratio.
        if not self.eval_mode and random.random() <= self.hp.description_free_g_ratio:
            ins_text = '"{}"'.format(transcript) if random.random() > 0.5 else '“{}”'.format(transcript)
        
        text_id = self.tokenizer(ins_text, return_tensors="pt").input_ids.squeeze()  # Batch size 1

        if text_id.shape[0] >= self.max_text_len:
            text_id = text_id[:self.max_text_len]
            # <eos> for mt5
            text_id[self.max_text_len-1] = 1

        return text_id, semantic_id, acoustic_id, lang_id, basename


    def collate_fn(self, batches: list) -> dict:
        """
        Collate function to process a batch of data samples, add offset & padding, concat sequence, generate segment ID, ...

        Args:
            batch (list): A list of tuples, where each tuple contains:
                - text_id (torch.Tensor)
                - semantic_id (np.ndarray)
                - acoustic_id (np.ndarray)
                - lang_id (int)
                - basename (str)

        Returns:
            dict: A dictionary containing batched data.
        """
        temp = []
        for text_id, semantic_id, acoustic_id, lang_id, basename in batches:
            semantic_id_len = semantic_id.shape[0]
            acoustic_id_len = acoustic_id.shape[0]

            # offset
            lang_id = lang_id + 1
            semantic_id = semantic_id + 1 + self.lang_num
            acoustic_id = acoustic_id + 1 + self.lang_num + self.hp.st_token_num
            
            if self.without_st:
                # to get 1st layer for AR, concat <BOS, Language ID, EOS1, AT 1st layer, EOS2>
                seq = np.asarray([self.bos_id] + [lang_id] + [self.eos_id] + list(acoustic_id[:, 0]) + [self.eos_id + 1])

                # to get full layers for NAR
                bos = np.stack([np.asarray([self.bos_id,])] * self.at_res_num, axis=1)
                lang = np.stack([np.asarray([lang_id,])] * self.at_res_num, axis=1) 
                eos1 = np.stack([np.asarray([self.eos_id,])] * self.at_res_num, axis=1) 
                full_seq = np.concatenate([bos, lang, eos1, acoustic_id, eos1 + 1], axis=0) 

                # get segment id
                segment_id = np.asarray([1] * (3) + [2] * (acoustic_id_len + 1))
            else:
                # to get 1st layer for AR, concat <BOS, Language ID, ST, EOS1, AT 1st layer, EOS2>
                seq = np.asarray([self.bos_id] + [lang_id] + list(semantic_id) + [self.eos_id] + list(acoustic_id[:, 0]) + [self.eos_id + 1])

                # to get full layers for NAR
                full_st_seq = np.stack([semantic_id] * self.at_res_num, axis=1) 
                bos = np.stack([np.asarray([self.bos_id,])] * self.at_res_num, axis=1) 
                lang = np.stack([np.asarray([lang_id,])] * self.at_res_num, axis=1)
                eos1 = np.stack([np.asarray([self.eos_id,])] * self.at_res_num, axis=1) 
                full_seq = np.concatenate([bos, lang, full_st_seq, eos1, acoustic_id, eos1 + 1], axis=0) 

                # get segment id
                segment_id = np.asarray([1] * (semantic_id_len + 3) + [2] * (acoustic_id_len + 1))

            # cut it off when reaching the maximum length.
            seq = seq[:self.max_len]
            full_seq = full_seq[:self.max_len, :]
            segment_id  = segment_id[:self.max_len]
            temp.append([text_id, seq, full_seq, segment_id, basename])

        # length padding
        basenames = []
        seqs = []
        full_seqs = []
        seq_lens = []
        segment_ids = []
        text_ids = []
        text_id_lens = []
        max_seq_len = max(seq.shape[0] for _, seq, _, _, _ in temp)
        
        for text_id, seq, full_seq, segment_id, basename in temp:
            seq_lens.append(seq.shape[0])
            text_id_lens.append(text_id.shape[0])

            seq = np.pad(seq,
                         (0, max_seq_len - seq.shape[0]),
                         mode='constant',
                         constant_values=0) 
            full_seq = np.pad(full_seq,
                              [(0, max_seq_len - full_seq.shape[0]), (0, 0)],
                              mode='constant',
                              constant_values=0) 
            segment_id = np.pad(segment_id,
                                (0, max_seq_len - segment_id.shape[0]),
                                mode='constant',
                                constant_values=0)
            text_id = np.pad(text_id, (0, self.max_text_len - text_id.shape[0]), mode='constant', constant_values=0)
            
            seqs.append(seq)
            full_seqs.append(full_seq)
            segment_ids.append(segment_id)
            text_ids.append(text_id)
            basenames.append(basename)         

        # to numpy, to torch
        seqs = torch.from_numpy(np.asarray(seqs))
        seq_lens = torch.from_numpy(np.asarray(seq_lens))
        full_seqs = torch.from_numpy(np.asarray(full_seqs))
        segment_ids = torch.from_numpy(np.asarray(segment_ids))
        text_ids = torch.from_numpy(np.asarray(text_ids))
        text_id_lens = torch.from_numpy(np.asarray(text_id_lens))

        return text_ids, text_id_lens, seqs, seq_lens, full_seqs, segment_ids, basenames


if __name__ == '__main__':
    from utils import get_config_from_file
    hp = get_config_from_file('./configs/train_ar_ft.yaml').hparams

    dataset = VoxDataset(meta_paths=['/zhangpai21/workspace/THU-TTS/data/wenetspeech/train_metadata_filter.json', ], hp=hp, is_check=False)
    # dataset = VoxDataset(meta_paths=hp.train_path, hp=hp, is_check=False)
    batches = [dataset.__getitem__(i) for i in tqdm(range(6))]
    x = dataset.collate_fn(batches)


