import os
import shutil
import argparse
import datetime
from pathlib import Path
from typing import Optional  

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model.nar import VoxInstructNAR
from utils.optimizer import get_optimizer, ScheduledOptim
from utils.dataset import VoxDataset
from utils.utils import (
    get_config_from_file, sequence_mask, compute_loss, accum_log, 
    calculate_model_params, cycle, to_device
)

class NARModelTrainer(nn.Module):
    """
    NARModelTrainer class for training the VoxInstructNAR model.
    """
    def __init__(
            self,
            config_path: str,
            save_path: str,
            restore_path: Optional[str] = None,
            log_type='tensorboard',
            logging_dir='logs',
            accelerate_kwargs: dict = dict(),
    ):
        """
        Initialize the NARModelTrainer class.

        Args:
            config_path (str): Path to the configuration file.
            save_path (str): Directory to save outputs.
            restore_path (Optional[str]): Path to restore model checkpoint.
            log_type (str): Type of logging to use.
            logging_dir (str): Directory for logging.
            accelerate_kwargs (dict): Additional arguments for Accelerator.
        """
        super(NARModelTrainer, self).__init__()
        hp = get_config_from_file(config_path).hparams
        
        self.save_path = save_path
        self.save_checkpoint_path = os.path.join(save_path, 'checkpoints')
        os.makedirs(self.save_checkpoint_path, exist_ok=True)

        # save config
        shutil.copyfile(config_path, os.path.join(self.save_path, 'config.yaml'))
        logging_dir = os.path.join(save_path, logging_dir)

        kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[kwargs],
                                       log_with=log_type,
                                       project_dir=logging_dir,
                                       gradient_accumulation_steps=hp.grad_accum_every,
                                       **accelerate_kwargs)
        self.accelerator.print(save_path)
        self.hp = hp
        
        model = VoxInstructNAR(hp=hp)
        calculate_model_params(model)
        self.model = model.to(self.accelerator.device)
        self.register_buffer('steps', torch.Tensor([0]))

        self.max_step = hp.max_step
        self.log_step = hp.log_step
        self.ckpt_step = hp.ckpt_step
        self.val_step = hp.val_step
        self.batch_size = hp.batch_size
        self.grad_accum_every = hp.grad_accum_every
        self.max_grad_norm = hp.max_grad_norm
        self.num_workers = hp.num_workers
           
        # optimizers
        self.optim = get_optimizer(
            self.model.parameters(),
            lr=hp.learning_rate, 
            wd=hp.weight_decay)
        self.scheduler_func = ScheduledOptim(
            warmup_steps=hp.warmup_step,
            num_gpu=self.accelerator.num_processes)
        self.scheduler = LambdaLR(self.optim, self.scheduler_func.get_lr_scale)

        if restore_path is not None:
            self.load(restore_path)

        self.trainset = VoxDataset(hp.train_path, hp=hp)
        self.valset = VoxDataset(hp.val_path, hp=hp, eval_mode=True)

        self.train_loader = DataLoader(self.trainset,
                            num_workers=self.num_workers,
                            shuffle=True,
                            batch_size=self.batch_size,
                            collate_fn=self.trainset.collate_fn)
        
        self.val_loader = DataLoader(self.valset,
                            num_workers=self.num_workers,
                            shuffle=False,
                            batch_size=self.batch_size,
                            collate_fn=self.valset.collate_fn)

        # prepare with accelerator
        (self.model, self.train_loader, self.optim, self.scheduler) = self.accelerator.prepare(self.model,
                                                    self.train_loader,
                                                    self.optim, 
                                                    self.scheduler)
    
        self.train_loader_iter = cycle(self.train_loader)
        self.val_loader_iter = cycle(self.val_loader)

        hps = {
            "num_train_steps": self.max_step,
            "batch_size": self.batch_size
        }
        self.accelerator.init_trackers(hp.name, config=hps)

    def save(self, path, steps, not_save_optim=False):

        if not_save_optim:
            ckpt = dict(model=self.accelerator.get_state_dict(self.model),
                   scheduler=self.scheduler.state_dict(),
                   steps=int(steps))
        else:
            ckpt = dict(model=self.accelerator.get_state_dict(self.model),
                   optim=self.optim.state_dict(),
                   scheduler=self.scheduler.state_dict(),
                   steps=int(steps))

        torch.save(ckpt, path)

    def load(self, path):
        path = Path(path)
        assert path is not None
        ckpt = torch.load(str(path), map_location='cpu')

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(ckpt['model'])
        
        if 'steps' in ckpt:
            self.steps = torch.Tensor([ckpt['steps']]) 
        if 'scheduler' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler']) 
        if 'optim' in ckpt:
            self.optim.load_state_dict(ckpt['optim']) 

    def print(self, msg):
        self.accelerator.print(msg)

    @property
    def unwrapped_model(self):
        return self.accelerator.unwrap_model(self.model)

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process
    
    def train_step(self):
        device = self.device
        self.model.train()
        logs = {}
        
        self.optim.zero_grad()

        start = datetime.datetime.now()
        steps = int(self.steps.item())

        for _ in range(self.grad_accum_every):
            with self.accelerator.accumulate(self.model):
                loaded_data = next(self.train_loader_iter)
                text_ids, text_id_lens, seqs, seq_lens, full_seqs, segment_ids, basenames = to_device(loaded_data, device=device)
                
                text_attn_mask = sequence_mask(text_id_lens, max_len=hp.max_text_len, device=device)
                seq_attn_mask = sequence_mask(seq_lens, max_len=None, device=device)
               
                logits, target_x, target_mask, act_mask_ratio = self.model(
                    full_seqs, 
                    segment_ids=segment_ids, 
                    attention_mask=seq_attn_mask,
                    text_ids=text_ids,   
                    text_attn_mask=text_attn_mask,     
                )
                
                # Compute loss and backpropagate
                loss, acc = compute_loss(logits, target_x, mask=target_mask, compute_acc=True, topk=(1,10))                   
                self.accelerator.backward(loss)
                
                # Clip gradients if max_grad_norm is set
                if self.max_grad_norm is not None:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    grad_norm = None

                accum_log(logs, {'loss': loss.item()})
                accum_log(logs, {'acc_top1': acc[0].item()})
                accum_log(logs, {'acc_top10': acc[1].item()})
                accum_log(logs, {'grad_norm': grad_norm.item()})
                accum_log(logs, {'samples': seqs.shape[0]})
                accum_log(logs, {'min_len': seq_lens.min().item()})
                accum_log(logs, {'max_len': seq_lens.max().item()})
                accum_log(logs, {'avg_mask_ratio': act_mask_ratio.mean().item()})

        # Step optimizer and scheduler after accumulation
        self.optim.step()
        self.scheduler.step()

        logs['loss'] = logs['loss'] / hp.grad_accum_every  
        logs['acc_top1'] = logs['acc_top1'] / hp.grad_accum_every  
        logs['acc_top10'] = logs['acc_top10'] / hp.grad_accum_every  
        logs['min_len'] = logs['min_len'] / hp.grad_accum_every  
        logs['max_len'] = logs['max_len'] / hp.grad_accum_every  
        logs['avg_mask_ratio'] = logs['avg_mask_ratio'] / hp.grad_accum_every  

        # log
        times = datetime.datetime.now() - start
        if steps % self.log_step == 0:         
            self.print("NAR training! step: {}, loss: {:.4f}, acc_top1: {:4f}, acc_top10: {:4f}, lr: {}, samples: {}, min_len:{}, max_len: {}, avg_mask_ratio: {}, grad_norm:{:4f}, times:{}\n".format(\
                        steps, logs['loss'], logs['acc_top1'], logs['acc_top10'], self.scheduler.get_last_lr(), logs['samples'], logs['min_len'], logs['max_len'], logs['avg_mask_ratio'], logs['grad_norm'], times))
        
            self.accelerator.log(
                {
                    "Train/loss": logs['loss'],
                    "Train/acc_top1": logs['acc_top1'],
                    "Train/acc_top10": logs['acc_top10'],    
                    "Train/lr": self.scheduler.get_last_lr()[0].item(),
                    "Train/avg_mask_ratio": logs["avg_mask_ratio"],
                    "Train/grad_norm": logs['grad_norm'],
                },
                step=steps)
            
        
        if self.is_main and self.val_step > 0 and steps % self.val_step == 0:
            eval_model = self.unwrapped_model
            eval_model.eval()
            val_loss = 0.
            token_cnt = 0
            all_acc_0 = 0.
            all_acc_1 = 0.
            for loaded_data in tqdm(self.val_loader):
                text_ids, text_id_lens, seqs, seq_lens, full_seqs, segment_ids, basenames = to_device(loaded_data, device=device)
                
                text_attn_mask = sequence_mask(text_id_lens, max_len=hp.max_text_len, device=device)
                seq_attn_mask = sequence_mask(seq_lens, max_len=None, device=device)

                with torch.no_grad(): 
                    logits, target_x, target_mask, avg_mask_ratio = self.model(
                        full_seqs, 
                        segment_ids=segment_ids, 
                        attention_mask=attention_mask,
                        text_ids = text_ids,   
                        text_attn_mask=text_attn_mask,   
                    )
                    # Compute loss
                    loss, acc = compute_loss(logits, target_x, mask=target_mask, compute_acc=True, topk=(1,10))   
              
                token_cnt += target_mask.sum().item()
                val_loss += loss.item() * target_mask.sum().item()
                all_acc_0 += acc[0].item() * target_mask.sum().item()
                all_acc_1 += acc[1].item() * target_mask.sum().item()
                       
            val_loss = val_loss / token_cnt
            all_acc_0 = all_acc_0 / token_cnt
            all_acc_1 = all_acc_1 / token_cnt
            self.accelerator.log({
                "Val/loss": val_loss,
                "Val/acc_top1": all_acc_0,
                "Val/acc_top10": all_acc_1,    
            }, step=steps)
            self.print("NAR validation! step: {}k, loss: {:.4f}, acc_top1: {:4f}, acc_top10: {:4f}\n".format(steps // 1000, val_loss, all_acc_0, all_acc_1))   
                     
        self.accelerator.wait_for_everyone()

        if self.is_main and steps % self.ckpt_step == 0:
            model_path = os.path.join(self.save_checkpoint_path, "{}k_ckpt.pyt".format(steps // 1000))
            self.save(model_path, steps, not_save_optim=True)
            self.print(f'{steps}: saving model at {model_path}')

        self.steps += 1
        return logs
        
 

    def train(self):
        while self.steps < self.max_step:
            logs = self.train_step()

        self.print('training complete')
        self.accelerator.end_training()


if __name__ == "__main__":
    torch.distributed.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=3600))
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, required=True, help='Config yaml')
    parser.add_argument('--save_path', type=str, required=True, help='Path to save checkpoints')
    parser.add_argument('--restore_path', default=None, type=str, help='restore checkpoints')

    args = parser.parse_args()
    hp = get_config_from_file(args.config).hparams

    trainer = NARModelTrainer(
        config_path=args.config,
        save_path=args.save_path,
        restore_path=args.restore_path,
    )

    trainer.train()

