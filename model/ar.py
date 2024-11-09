import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, MT5EncoderModel
from peft import LoraConfig, get_peft_model
import random


class VoxInstructAR(nn.Module):
    def __init__(self, hp):
        super(VoxInstructAR, self).__init__()
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
        self.model = LlamaForCausalLM(config=lm_config)

        self.embed_tokens = nn.Embedding(lm_config.vocab_size, hp.hidden_dim, 0)
        nn.init.trunc_normal_(self.embed_tokens.weight, std=0.02)
        nn.init.constant_(self.embed_tokens.weight[0], 0.0)
        self.embed_segments = nn.Embedding(3, hp.hidden_dim)
        
        self.prompt_fc = nn.Linear(hp.mt5_out_dim, hp.hidden_dim)
        self.text_free_g_ratio = hp.text_free_g_ratio
        self.st_free_g_ratio = hp.st_free_g_ratio
        

    def forward(self, input_ids, segment_ids, attention_mask=None, text_ids=None, text_attn_mask=None):
        device = input_ids.device
        batch_size = input_ids.shape[0]

        # get text embeddings from pre-trained mt5 encoder 
        t5_outputs = self.t5_model(input_ids=text_ids, attention_mask=text_attn_mask)
        text_encode = self.prompt_fc(t5_outputs.last_hidden_state)
        text_encode = torch.where(
            (text_attn_mask > 0).unsqueeze(-1), 
            text_encode, 
            torch.zeros_like(text_encode, device=device)
        )

        # mask text_encode part
        prob_1 = torch.rand((batch_size,), device=device)
        text_free_cond = (prob_1 < self.text_free_g_ratio).unsqueeze(-1).unsqueeze(-1).expand_as(text_encode)
        text_encode = torch.where(
            text_free_cond, 
            torch.zeros_like(text_encode, device=device), 
            text_encode
        )
        
        # mask semantic token part
        prob_2 = torch.rand((batch_size,), device=device)
        st_free_cond = (segment_ids == 1) & (prob_2 < self.st_free_g_ratio).unsqueeze(-1).expand_as(segment_ids)
        input_ids = torch.where(st_free_cond, torch.zeros_like(input_ids, device=device), input_ids)
        
        # concat text encoding and inputs (st & at)
        extend_inputs_embeds = torch.cat([text_encode, self.embed_tokens(input_ids)], dim=1)
        extend_segment_ids = torch.cat([torch.zeros_like(text_ids, device=device), segment_ids], dim=1)
        extend_inputs_embeds += self.embed_segments(extend_segment_ids)
        extend_attention_mask = torch.cat([torch.ones_like(text_attn_mask, device=device), attention_mask], dim=1)

        outputs = self.model(
                inputs_embeds=extend_inputs_embeds,
                attention_mask=extend_attention_mask,
            )
        return outputs


    def predict(self, input_ids, segment_ids, attention_mask=None, text_ids=None, text_attn_mask=None, past_key_values=None, text_encode=None, mask_st=False):
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if text_encode is None:
            t5_outputs = self.t5_model(input_ids=text_ids, attention_mask=text_attn_mask)
            text_encode = self.prompt_fc(t5_outputs.last_hidden_state)
            text_encode = torch.where(
                (text_attn_mask > 0).unsqueeze(-1), 
                text_encode, 
                torch.zeros_like(text_encode, device=device)
            )

        if mask_st:
            st_free_cond = (segment_ids == 1)
            input_ids = torch.where(st_free_cond, torch.zeros_like(input_ids), input_ids)

        extend_inputs_embeds = torch.cat([text_encode, self.embed_tokens(input_ids)], dim=1)
        extend_segment_ids = torch.cat([torch.zeros_like(text_ids, device=device), segment_ids], dim=1)
        extend_inputs_embeds = extend_inputs_embeds + self.embed_segments(extend_segment_ids)
        
        # import pdb; pdb.set_trace()
        if past_key_values is None:
            outputs = self.model(
                inputs_embeds=extend_inputs_embeds,
                use_cache=True,
            )
        else:
            outputs = self.model(
                inputs_embeds=extend_inputs_embeds[:, -1:, :],
                use_cache=True,
                past_key_values=past_key_values,
            )

        return outputs, text_encode
