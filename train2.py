import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from transformer2 import Transformer
from datasets import load_dataset
import sentencepiece as spm
import tqdm
import constants
import wandb
import os
import re

# start a new wandb run to track this script
if constants.WANDB_ON:
  wandb.init(
      # set the wandb project where this run will be logged
      project="Transformer",
      
      # track hyperparameters and run metadata
      config={
      "learning_rate": constants.LEARNING_RATE,
      "dimensions": constants.DIMENSIONS,
      "dataset": constants.DATASET,
      "vocab_size": constants.VOCAB_SIZE,
      "epochs": constants.NUM_OF_EPOCHS,
      "num_heads" : constants.NUM_HEADS,
      "num_layers" : constants.NUM_LAYERS,
      "d_ff" : constants.D_FF,
      "max_seq_length" : constants.MAX_SEQ_LENGTH,
      "dropout" : constants.DROPOUT
      }
  )

class TinyStoriesData(torch.utils.data.Dataset):
  def __init__(self, name, mode, max_seq_length):
    self.dataset = load_dataset(name,split=mode)
    self.sp = spm.SentencePieceProcessor(model_file='tinystorycustom.model')
    self.data = []
    self.create_data(max_seq_length)

  def create_data(self, max_seq_length):
    encoded = self.sp.encode(self.dataset['text'], add_bos=True, add_eos=True)
    padded = encoded + [3] * (max_seq_length - len(encoded))
    self.data.append(padded)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return torch.tensor(self.data[idx])

ds = TinyStoriesData("roneneldan/TinyStories", "train", constants.MAX_SEQ_LENGTH)

dl = torch.utils.data.DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True)



# Generate random sample data
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

transformer = Transformer(constants.VOCAB_SIZE, constants.DIMENSIONS, constants.NUM_HEADS, constants.NUM_LAYERS, constants.D_FF, constants.MAX_SEQ_LENGTH, constants.DROPOUT)
criterion = nn.CrossEntropyLoss(ignore_index=3) # sentencepiece pad_id = 3
optimizer = optim.Adam(transformer.parameters(), lr=constants.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

transformer.train()

# for epoch in range(100):
#     optimizer.zero_grad()
#     output = transformer(tgt_data[:, :-1])
#     loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

def find_latest_epoch_file(path='./'):
    epoch_files = [f for f in os.listdir(path) if re.match(r'transformer_epoch_\d+\.pt', f)]
    if epoch_files:
        # Extracting epoch numbers from the files and finding the max
        latest_epoch = max([int(f.split('_')[2].split('.')[0]) for f in epoch_files])
        return latest_epoch, f"./transformer_epoch_{latest_epoch}.pt"
    else:
        return 0, None

# Function to load the latest epoch file if it exists
def load_latest_checkpoint(model, path='./'):
    latest_epoch, latest_file = find_latest_epoch_file(path)
    if latest_file:
        print(f"Resuming training from epoch {latest_epoch+1}")
        model.load_state_dict(torch.load(latest_file))
    else:
        print("No checkpoint found, starting from beginning")
    return latest_epoch

start_epoch = load_latest_checkpoint(transformer)

for epoch in range(start_epoch, constants.NUM_OF_EPOCHS):
  total_loss = 0
  for tgt_data in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}", unit="batch"):
    optimizer.zero_grad()
    output = transformer(tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, constants.VOCAB_SIZE), tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}, Loss: {total_loss}")
  torch.save(transformer.state_dict(), f"./transformer_epoch_{epoch+1}.pt")
  if constants.WANDB_ON:
    wandb.log({"acc": 2, "total_loss": total_loss})
  
if constants.WANDB_ON:
  wandb.finish()


