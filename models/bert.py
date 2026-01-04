'''
* @name: bert.py
* @description: Functions of BERT for Chinese.
'''


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, RobertaModel, RobertaTokenizer

__all__ = ['BertTextEncoder']

# 新增中文预训练模型映射
TRANSFORMERS_MAP = {
    'bert': (BertModel, BertTokenizer),
    'roberta': (RobertaModel, RobertaTokenizer),
    'bert-chinese': (BertModel, BertTokenizer),  # 中文bert-base-chinese模型
    'roberta-chinese': (RobertaModel, RobertaTokenizer),  # 中文RoBERTa模型
}


class BertTextEncoder(nn.Module):
    def __init__(self, use_finetune=False, transformers='bert-chinese',
                 pretrained='bert-base-chinese'):  # 默认使用中文BERT
        super().__init__()
        tokenizer_class = TRANSFORMERS_MAP[transformers][1]
        model_class = TRANSFORMERS_MAP[transformers][0]
        self.tokenizer = tokenizer_class.from_pretrained(pretrained, use_safetensors=True,ignore_mismatched_sizes=True)
        self.model = model_class.from_pretrained(pretrained, use_safetensors=True,ignore_mismatched_sizes=True)
        self.use_finetune = use_finetune

    def get_tokenizer(self):
        return self.tokenizer

    def forward(self, text):
        """
        text: (batch_size, 3, seq_len)
        3: input_ids, input_mask, segment_ids
        input_ids: input_ids,
        input_mask: attention_mask,
        segment_ids: token_type_ids
        """
        input_ids, input_mask, segment_ids = text[:,0,:].long(), text[:,1,:].float(), text[:,2,:].long()
        if self.use_finetune:
            last_hidden_states = self.model(input_ids=input_ids,
                                            attention_mask=input_mask,
                                            token_type_ids=segment_ids)[0]
        else:
            with torch.no_grad():
                last_hidden_states = self.model(input_ids=input_ids,
                                                attention_mask=input_mask,
                                                token_type_ids=segment_ids)[0]
        return last_hidden_states  # (batch_size, seq_len, d_model)