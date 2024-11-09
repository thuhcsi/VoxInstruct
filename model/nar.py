import math
import torch
import torch.nn.functional as F
from torch import nn
from transformers import MT5EncoderModel, LlamaConfig, LlamaModel
from peft import LoraConfig, get_peft_model
import random

class VoxInstructNAR(nn.Module):
    def __init__(self, hp):
        super(VoxInstructNAR, self).__init__()
        self.hp = hp

        self.t5_model = MT5EncoderModel.from_pretrained(hp.mt5_path)
        for p in self.t5_model.parameters():
            p.requires_grad = False

        if hp.apply_lora:
            lora_config = LoraConfig(
                r=hp.lora_r,
                lora_alpha=hp.lora_alpha,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                bias="none",
                modules_to_save=[],
            )
            self.t5_model = get_peft_model(self.t5_model, lora_config)
            self.lora_config = lora_config
        
        lm_config = LlamaConfig(
            vocab_size=1+hp.lang_num+hp.st_token_num+hp.at_token_num+3, # padding=0, lang index, semantic index, codec index, BOS, EOS 1, EOS 2
            hidden_size=hp.hidden_dim, 
            intermediate_size=hp.ffn_inner_dim,
            num_hidden_layers=hp.num_layers, 
            num_attention_heads=hp.num_heads, 
            hidden_act=hp.ffn_act_func,
            max_position_embeddings=hp.max_len+hp.max_text_len, 
            pad_token_id=0,
            bos_token_id=hp.bos_id,
            eos_token_id=hp.eos_id,
            attention_dropout=hp.attn_dropout_p,
            _attn_implementation="flash_attention_2"
        )
        self.model = LlamaModel(config=lm_config)
        # Non-causal for NAR model
        for layer in self.model.layers:
            layer.self_attn.is_causal = False
        
        self.all_embed_tokens = nn.ModuleList(
            [nn.Embedding(lm_config.vocab_size, embedding_dim=hp.hidden_dim, padding_idx=0) for _ in range(hp.at_res_num)]
        )
        for embed in self.all_embed_tokens:
            nn.init.trunc_normal_(embed.weight, std=0.02)
            nn.init.constant_(embed.weight[0], 0.0)
        self.embed_segments = nn.Embedding(3, hp.hidden_dim)
        self.embed_res = nn.Embedding(hp.at_res_num - 1, hp.hidden_dim)
   
        self.out_proj = nn.ModuleList(
            [nn.Linear(hp.hidden_dim, hp.at_token_num + 3, bias=False) for _ in range(hp.at_res_num - 1)]
        )
        self.prompt_fc = nn.Linear(hp.mt5_out_dim, hp.hidden_dim)
        self.text_free_g_ratio = hp.text_free_g_ratio
        self.at_free_g_ratio = hp.at_free_g_ratio

    def random_masking(self, st_lens, at_lens):
        """
        Ref: SoundStorm (https://arxiv.org/abs/2305.09636)
        Ref: https://github.com/facebookresearch/mae/blob/main/models_mae.py#L123
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        """
        N = st_lens.shape[0] 
        seq_lens = st_lens + at_lens
        L = seq_lens.max()
        device = st_lens.device

        # reserve prompt u~(0, T-1) 
        if random.random() < self.at_free_g_ratio:
            unmask_ratio = torch.zeros(size=[N,], device=device)
        else:
            unmask_ratio = torch.rand(size=[N,], device=device)
        
        unmask_lens = st_lens + (unmask_ratio * at_lens).floor().long() 
        
        # mask in valid part
        mask_prompt = (unmask_lens).unsqueeze(1) > torch.arange(L, device=device).unsqueeze(0) # [N, 1] > [1, L] = [N, L]
        
        if self.hp.mask_strategy == "cosine":
            mask_ratio = torch.cos(math.pi / 2 * torch.rand(size=(N,), device=device))
        elif self.hp.mask_strategy == "full":
            mast_ratio = torch.ones(size=[N,], device=device)
        
        mask_len = ((seq_lens - unmask_lens) * mask_ratio).floor().clamp(1).long()
        
        # sort ascending noise for each sample: small is keep, large is remove
        noise = torch.rand(N, L, device=device)  
        for i in range(N):
            noise[i, unmask_lens[i]:seq_lens[i]] -= 1
        ids_shuffle = torch.argsort(noise, dim=1) 
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=device)
        for i in range(N):
            mask[i, :mask_len[i]] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)    

        return mask.bool(), ids_restore, mask_prompt, mask_ratio
    

    def forward(self, input_ids, segment_ids, attention_mask=None, text_ids=None, text_attn_mask=None):
        '''
        NAR training forward process.
        To perform multi-step iterative decoding in inference, use random mask strategy like SoundStorm.
        '''
        # input_ids: [B, t, n_codebook] format [repeated st, at_i]
        x = input_ids.transpose(1, 2)
        batch_size, n_codebook, t = x.size()
        assert n_codebook == self.hp.at_res_num, f"n_codebook: {n_codebook} != {self.hp.at_res_num}"
        device = x.device

        # get text embeddings from pre-trained mt5 encoder 
        t5_outputs = self.t5_model(input_ids=text_ids, attention_mask=text_attn_mask)
        text_encode = self.prompt_fc(t5_outputs.last_hidden_state)
        text_encode = torch.where(
            (text_attn_mask > 0).unsqueeze(-1), 
            text_encode, 
            torch.zeros_like(text_encode, device=device)
        )

        prob = torch.rand((batch_size,), device=device)
        text_free_cond = (prob < self.text_free_g_ratio).unsqueeze(-1).unsqueeze(-1).expand_as(text_encode)
        text_encode = torch.where(
            text_free_cond, 
            torch.zeros_like(text_encode, device=device), 
            text_encode
        )
        
        st_lens = (segment_ids == 1).sum(dim=1)
        at_lens = (segment_ids == 2).sum(dim=1)
        
        # random mask previous residual layers of 1~Q
        layer_index = torch.randint(low=1, high=n_codebook, size=[batch_size,], device=device)
        mask1 = layer_index.unsqueeze(1) > torch.arange(n_codebook, device=device).unsqueeze(0) # [b, 1] > [1, n_codebook] = [b, n_codebook]
        
        # random mask n_token in Q-th residual layer
        mask2, _, mask_prompt, mask_ratio = self.random_masking(st_lens, at_lens) # [b, t]
   
        # total mask for x, and get mask_x 
        mask = (mask1.unsqueeze(2) + mask_prompt.unsqueeze(1))
        mask_x = torch.where(mask, x, torch.zeros_like(x)) # [b, n_codebook, t]
        for i, ind in enumerate(layer_index):
            mask[i, ind, :] = mask2[i]
            mask_x[i, ind, :] = torch.where(mask2[i], mask_x[i, ind, :], torch.zeros_like(mask_x[i, ind, :]))

        # sum all embeddings (layer info, st, at) to get inputs_embeds
        inputs_embeds = self.embed_res(layer_index - 1).unsqueeze(1) 
        for i in range(n_codebook):
            inputs_embeds = self.all_embed_tokens[i](mask_x[:, i, :]) + inputs_embeds

        # concat text encoding and inputs (st & at)
        extend_inputs_embeds = torch.cat([text_encode, inputs_embeds], dim=1)
        extend_segment_ids = torch.cat([torch.zeros_like(text_ids, device=device), segment_ids], dim=1)
        extend_inputs_embeds += self.embed_segments(extend_segment_ids)
        extend_attention_mask = torch.cat([torch.ones_like(text_attn_mask, device=device), attention_mask], dim=1)

        hidden_outputs = self.model(
                inputs_embeds=extend_inputs_embeds,
                attention_mask=extend_attention_mask,
            ).last_hidden_state[:, self.hp.max_text_len:, :] 
        
        # get corrseponding head logits and target token
        target_x = []
        logits = []
        for i, ind in enumerate(layer_index):
            target_x.append(x[i, ind, :]) # [b, n_codebook, t] -> b * [t]
            logits.append(self.out_proj[ind - 1](hidden_outputs[i]))
        logits = torch.stack(logits, dim=0)
        target_x = torch.stack(target_x, dim=0)   
        target_x = (target_x - 1 - self.hp.st_token_num - self.hp.lang_num).clamp(0) # remove padding, lang_offset & semantic offset, just acoustic part (0-1023)
        
        target_mask = attention_mask * (~mask2)
        
        act_mask_ratio = target_mask.sum(dim=1).float() / at_lens
        return logits, target_x, target_mask, act_mask_ratio

    def predict(self, x, at_prompt_lens, segment_ids, text_ids=None, text_attn_mask=None, layer_index=1, iter_step=1):
        '''
        Perform iterative decoding to predict the residual layers of acoustic tokens.
        '''
        batch_size, n_codebook, t = x.size()
        assert n_codebook == self.hp.at_res_num, f"n_codebook: {n_codebook} != {self.hp.at_res_num}"
        device = x.device

        # get text embeddings from pre-trained mt5 encoder 
        t5_outputs = self.t5_model(input_ids=text_ids, attention_mask=text_attn_mask)
        text_encode = self.prompt_fc(t5_outputs.last_hidden_state)
        text_encode = torch.where(
            (text_attn_mask > 0).unsqueeze(-1), 
            text_encode, 
            torch.zeros_like(text_encode, device=device)
        )

        st_lens = (segment_ids == 1).sum(dim=1)
        at_lens = (segment_ids == 2).sum(dim=1)

        if isinstance(layer_index, int):
            layer_index = torch.LongTensor([layer_index,]).repeat(batch_size,).to(device)

        # mask_to_predict: 1 for masking, [1, t]
        mask_to_predict = (st_lens + at_prompt_lens).unsqueeze(1) <= torch.arange(t, device=device).unsqueeze(0) # [b, 1] > [1, t] = [b, t]
        num_to_mask = int(torch.sum(mask_to_predict))
        num_to_mask_pre = 0

        # iterative decoding, <iter_step> times
        for k in range(iter_step):
            # Determine the number of tokens to be masked in this step using a cosine strategy
            num_to_mask_step = math.ceil(num_to_mask * (1 - math.cos(math.pi / 2 * (k+1) / iter_step)) - num_to_mask_pre)  

            inputs_embeds = self.embed_res(layer_index - 1).unsqueeze(1) 
            for i in range(n_codebook):
                inputs_embeds = self.all_embed_tokens[i](x[:, i, :]) + inputs_embeds

            extend_inputs_embeds = torch.cat([text_encode, inputs_embeds], dim=1)
            extend_segment_ids = torch.cat([torch.zeros_like(text_ids, device=device), segment_ids], dim=1)
            extend_inputs_embeds += self.embed_segments(extend_segment_ids)

            hidden_outputs = self.model(
                    inputs_embeds=extend_inputs_embeds,
                ).last_hidden_state[:, self.hp.max_text_len:, :] 

            logits = self.out_proj[layer_index - 1](hidden_outputs)

            # select tokens by confidence score at each iterative decoding step
            logits_confid = F.softmax(logits, dim=-1).max(dim=-1).values.detach()
            pred_logits_confidence = logits_confid * mask_to_predict.float()
            vals, inds = torch.sort(pred_logits_confidence, descending=True)
            selected_inds = inds[:, :num_to_mask_step] 

            # add offset to pred acoustic tokens
            pred_toks = logits.argmax(dim=-1) + self.hp.st_token_num + self.hp.lang_num + 1

            x[0, layer_index] = x[0, layer_index].scatter(1, selected_inds, pred_toks[0, inds])
            mask_to_predict.scatter_(1, selected_inds, torch.zeros_like(mask_to_predict, device=device))
            num_to_mask_pre += num_to_mask_step

        return x
    
