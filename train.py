import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from transformer import Transformer
import dataset
import sentencepiece as spm
import tqdm
import constants
import wandb
import utilities

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

device = utilities.getDevice()
print(f"Device = {device}")

ds = dataset.TinyStoriesData("roneneldan/TinyStories", f"train[:{constants.DATASET_PERCENTAGE}%]", constants.MAX_SEQ_LENGTH)
dl = torch.utils.data.DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True, collate_fn=ds.collate_function)

# Generate random sample data
# tgt_data = torch.randint(1, tgt_vocab_size, (64, max_seq_length))  # (batch_size, seq_length)

transformer = Transformer(constants.VOCAB_SIZE, constants.DIMENSIONS, constants.NUM_HEADS, constants.NUM_LAYERS, constants.D_FF, constants.MAX_SEQ_LENGTH, constants.DROPOUT).to(device)
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

start_epoch = utilities.load_latest_checkpoint(transformer)

for epoch in range(start_epoch, constants.NUM_OF_EPOCHS):
  total_loss = 0
  for tgt_data in tqdm.tqdm(dl, desc=f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}", unit="batch"):
    optimizer.zero_grad()
    output = transformer(tgt_data[:, :-1].to(device))
    loss = criterion(output.to(device).contiguous().view(-1, constants.VOCAB_SIZE), tgt_data[:, 1:].to(device).contiguous().view(-1))
    loss.backward()
    optimizer.step()
    total_loss += loss.item()
  print(f"Epoch {epoch+1}/{constants.NUM_OF_EPOCHS}, Loss: {total_loss}")
  torch.save(transformer.state_dict(), f"./transformer_epoch_{epoch+1}.pt")
  if constants.WANDB_ON:
    wandb.log({"acc": 2, "total_loss": total_loss})
  
if constants.WANDB_ON:
  wandb.finish()


