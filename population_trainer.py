from ray import train, tune
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
import os
import re
import utilities

def train_transformer(config):
    device = utilities.getDevice()
    print(f"Device = {device}")

    #load data
    ds = dataset.TinyStoriesData("roneneldan/TinyStories", "train[:1%]", constants.MAX_SEQ_LENGTH)
    dl = torch.utils.data.DataLoader(ds, batch_size=constants.BATCH_SIZE, shuffle=True)

    # Instantiate Transformer with the hyperparameters from the configuration
    transformer = Transformer(
        vocab_size=config["vocab_size"],
        dimensions=config["dimensions"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_length=config["max_seq_length"],
        dropout=config["dropout"]
    ).to(utilities.getDevice())

    #implement transformer 
    criterion = nn.CrossEntropyLoss(ignore_index=3) # sentencepiece pad_id = 3 #calculate loss 
    optimizer = optim.Adam(transformer.parameters(), lr=constants.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    transformer.train()

    for epoch in range(0,constants.NUM_OF_EPOCHS):
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
        average_loss = total_loss / len(dl)
        
        if constants.WANDB_ON:
            wandb.log({"average_loss": average_loss, "total_loss": total_loss})
  
    if constants.WANDB_ON:
        wandb.finish()
    
    return average_loss


search_space = {

    "vocab_size": constants.VOCAB_SIZE,
    "learning_rate": tune.loguniform(1e-5, 1e-1),
    "dimensions": tune.choice([64,128,256, 512]),
    "num_heads": tune.choice([2,4, 8]),
    "num_layers": tune.choice([1,2,4,6]),
    "d_ff": tune.choice([1024, 2048]),
    "max_seq_length": constants.MAX_SEQ_LENGTH,
    "dropout": tune.uniform(0.1, 0.5)
}

pbt_scheduler = tune.schedulers.PopulationBasedTraining(
    time_attr="training_iteration",
    metric="loss",
    mode="min",
    perturbation_interval=3,
    hyperparam_mutations=search_space
)

pbt_scheduler = tune.schedulers.PopulationBasedTraining(
    time_attr="training_iteration",
    metric="loss",
    mode="min",
    perturbation_interval=3,
    hyperparam_mutations=search_space
)


analysis = tune.run(
    train_transformer,
    config=search_space,
    num_samples= 4,
    scheduler=pbt_scheduler,
    stop={"training_iteration": 20}  # Adjust num_epochs as needed
)