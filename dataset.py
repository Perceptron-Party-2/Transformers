import torch
from datasets import load_dataset
import sentencepiece as spm

class TinyStoriesData(torch.utils.data.Dataset):
  def __init__(self, name, mode, max_seq_length):
    self.dataset = load_dataset(name,split=mode)
    self.sp = spm.SentencePieceProcessor(model_file='tinystorycustom.model')
    # self.data = []
    # self.create_data(max_seq_length)

  # def create_data(self, max_seq_length):
  #   train_data = self.dataset
  #   max = 0
  #   for record in train_data: 
  #     sentence = record["text"]  # Adjust the key based on the actual structure of your dataset
  #     encoded =self.sp.encode_as_ids(sentence, add_bos=True, add_eos=True)
  #     enc_length = len(encoded)
  #     if enc_length > max:
  #       max = enc_length
  #     truncated = encoded[0:max_seq_length]
  #     padded = truncated + [3] * (max_seq_length - len(truncated))
  #     self.data.append(padded)
  #   print(f"Longest encoding = {max}")
  #   print(f"Length of data created: {len(self.data)}")
    

  def __len__(self):
    return len(self.dataset)

  def __getitem__(self, idx):
    sentence = self.dataset[idx]["text"]
    encoded =self.sp.encode_as_ids(sentence, add_bos=True, add_eos=True)
    return torch.tensor(encoded)
  
  def collate_function(self, batch):
    return torch.nn.utils.rnn.pad_sequence([item for item in batch], batch_first=True, padding_value=3)
