import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from transformer2 import Transformer
from datasets import load_dataset
import sentencepiece as spm
import tqdm


tgt_vocab_size = 5000
d_model = 512
num_heads = 8
num_layers = 6
d_ff = 2048
max_seq_length = 1200
dropout = 0.1

class TinyStoriesData(torch.utils.data.Dataset):
  def __init__(self, name, mode, max_seq_length):
    self.dataset = load_dataset(name)
    self.sp = spm.SentencePieceProcessor(model_file='/home/louis/projects/week3/tinystorycustom.model')
    self.data = []
    self.create_data(max_seq_length)
    

  def create_data(self, max_seq_length):
    train_data = self.dataset["train"]
    for record in train_data: 
      sentence = record["text"]  # Adjust the key based on the actual structure of your dataset
      encoded =self.sp.encode_as_ids(sentence, add_bos=True, add_eos=True)
      padded = encoded + [3] * (max_seq_length - len(encoded))
      self.data.append(padded)

    

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return torch.tensor(self.data[idx])

ds = TinyStoriesData("roneneldan/TinyStories", "train", max_seq_length)

dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)



# Generate random sample data
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)



transformer = Transformer(tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss(ignore_index=3) # sentencepiece pad_id = 3
optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

# for epoch in range(100):
#     optimizer.zero_grad()
#     output = transformer(tgt_data[:, :-1])
#     loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

for epoch in range(10):
  total_loss = 0
  for tgt_data in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/10", unit="batch"):
    optimizer.zero_grad()
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}/10, Loss: {total_loss}")
  torch.save(transformer.state_dict(), f"./transformer_epoch_{epoch+1}.pt")
