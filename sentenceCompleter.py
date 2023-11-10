import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from transformer2 import Transformer
import sentencepiece as spm
import constants
import os
import re


# Generate random sample data
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

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

def generate(text):

  sp = spm.SentencePieceProcessor(model_file='tinystorycustom.model')
  encodedSentence = torch.tensor(sp.encode_as_ids(text, add_bos=True)).long().unsqueeze(0)
  transformer = Transformer(constants.VOCAB_SIZE, constants.DIMENSIONS, constants.NUM_HEADS, constants.NUM_LAYERS, constants.D_FF, constants.MAX_SEQ_LENGTH, constants.DROPOUT)
  transformer.eval()
  load_latest_checkpoint(transformer)
  
  for _ in range(20):
    with torch.no_grad():
      logits = transformer(encodedSentence)
      logits = logits[:, -1, :] / 1.0
      probs = torch.nn.functional.softmax(logits, dim=-1)
      next = torch.multinomial(probs, num_samples=1)
      if next.item() == 2: break
      encodedSentence = torch.cat([encodedSentence, next], dim=1)

  output = sp.decode(encodedSentence.tolist()[0])
  print(f"{text} - {output}")
  return { "story" : output } 

