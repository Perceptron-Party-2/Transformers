from datasets import load_dataset
import sentencepiece as spm
dataset = load_dataset("roneneldan/TinyStories")

model_prefix = "tinystorycustom"
vocab_size = 16000 

text_file = 'tinystories'

with open(text_file,'w',encoding='utf-8') as file:
    for text in dataset['train']['text']:
        file.write(text)

# Train the SentencePiece model
spm.SentencePieceTrainer.train(
    input=text_file,  # Pass the text data from the DataFrame
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    pad_id=3
)
