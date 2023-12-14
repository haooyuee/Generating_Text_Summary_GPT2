import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import add_special_tokens


class GPT21024Dataset(Dataset):

    def __init__(self, root_dir, ids_file, mode='train',length=None):
        self.root_dir = root_dir
        self.tokenizer = add_special_tokens()

        with open(ids_file,'r') as f:
            if mode=='train':
                self.idxs = np.array(json.load(f)['train_ids'])
            elif mode=='valid':
                self.idxs = np.array(json.load(f)['valid_ids'])
            elif mode=='test':
                self.idxs = np.array(json.load(f)['test_ids'])

            self.idxs = self.idxs -min(self.idxs)
        
        #self.idxs = os.listdir(root_dir)
        self.mode = mode
        if len == None:
            self.len = len(self.idxs)
        else:
            self.len = length

    def __len__(self):
        return self.len

    def __getitem__(self,idx):

        if self.mode=='valid':
            idx = self.idxs[idx]
        elif self.mode=='test':
            idx = self.idxs[idx]   # assuming valid and test set of same sizes
        else:
            idx = self.idxs[idx]
        file_name = os.path.join(self.root_dir,str(idx)+".json")
        #file_name = os.path.join(self.root_dir,str(idx))
        with open(file_name,'r') as f:
              data = json.load(f)
        text = self.tokenizer.encode(self.tokenizer.pad_token)*1024
        content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
        text[:len(content)] = content
        text = torch.tensor(text)
        sample = {'article': text, 'sum_idx': len(data['article'])}
        return sample
    
class GPT21024Dataset_new(Dataset):
    def __init__(self, file_path, length=None):
        """
        初始化数据集
        Args:
            file_path (str): CSV文件的路径.
            mode (str): 数据集模式（'train', 'valid', 'test'）.
            length (int): 限制数据集的大小，None表示使用全部数据.
        """
        self.data = pd.read_csv(file_path, sep='\t')
        if length is not None:
            self.data = self.data.iloc[:length]
        self.tokenizer = add_special_tokens()
        self.max_length = 1024

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        source_text = row['document']
        target_text = row['summary']

        # 对文档和摘要进行编码
        encoded_source = self.tokenizer.encode(source_text)
        encoded_target = self.tokenizer.encode(target_text)

        # 计算截断长度
        total_len = len(encoded_source) + len(encoded_target) + 1  # +1 是因为分隔符
        if total_len > self.max_length:
            # 超过最大长度时，需要截断
            # 优先保留摘要的完整性，因此先截断文档
            excess_len = total_len - self.max_length
            encoded_source = encoded_source[:-excess_len]
            print(len(encoded_source))

        # 添加分隔符并截断或填充
        text = self.tokenizer.encode(self.tokenizer.pad_token) * self.max_length
        content = encoded_source + self.tokenizer.encode(self.tokenizer.sep_token) + encoded_target
        content = content[:self.max_length]  # 确保不超过最大长度
        text[:len(content)] = content
        text = torch.tensor(text)
        print("sample len -->: {}".format(len(content)))
        print("sample len -->: {}".format(len(text)))
        sample = {'article': text, 'sum_idx': len(encoded_source)}
        return sample

        # # 添加分隔符并截断或填充
        # text = self.tokenizer.encode(self.tokenizer.pad_token)*1024
        # content = encoded_source + self.tokenizer.encode(self.tokenizer.sep_token) + encoded_target
        # text[:len(content)] = content
        # text = torch.tensor(text)
        # sample = {'article': text, 'sum_idx': len(encoded_source)}
        # return sample