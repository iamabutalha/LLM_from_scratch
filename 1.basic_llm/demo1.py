import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_block import Block, MultiHeadAttention, FeedForward

print(f"PyTorch version: {torch.__version__}")

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Talha",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
# print(text)

words = list(set(text.split()))

# print(words)


vocab_size = len(words)
print(f"Vocab size: {vocab_size}")


word2idx = {w: i for i, w in enumerate(words)}
# print(word2idx)

data = torch.tensor([word2idx[w] for w in text.split()], dtype=torch.long)

print(data)
print(len(data))

'''
tensor([ 3, 39, 16,  5, 30, 33,  8, 37, 19, 28,  6, 33, 20, 10, 19, 21, 33,  8,
        25,  9,  1,  5,  2, 33, 31, 19, 17,  4, 12, 33,  8, 23, 19, 13, 36, 33,
         7, 22, 41, 32, 14, 27, 37, 33, 24, 19, 20, 40, 11, 33, 35, 29, 34, 14,
        18, 33,  0, 26,  8, 38, 15, 33])


hello - 3 
'''

block_size = 6 #how many words LLM can see in the past (context length)
embedding_size = 32 #embedding size(dimension) for each word
n_heads = 4 #number of attention heads
n_layers = 2 #number of transformer blocks
lr = 1e-3 #learning rate
epochs = 1500 #number of epochs to train

def get_batch(batch_size=16):
    # 16 examples (sentences sequences) between 0-55 [3,12,3,21.......] and every example has 6 words sentence sequence
    # 6 words become a sequcence [3,4,5,6,7,8]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    # x = [token_12, token_13, token_14, token_15, token_16, token_17]
    # y = [token_13, token_14, token_15, token_16, token_17, token_18]
    return x, y


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size) # vocab_size = 42, embedding_size = 32


        self.position_embedding_table = nn.Embedding(block_size, embedding_size) # 6, 32 provides positional information to the model, the position of each token in the input sequence is encoded as a vector and added to the token embeddings


        self.blocks = nn.Sequential(*[Block(embedding_size, n_heads) for _ in range(n_layers)])

        self.ln_f = nn.LayerNorm(embedding_size)
        self.lm_head = nn.Linear(embedding_size, vocab_size) # will convert the final hidden state of the model into a probability distribution over the vocabulary 32 dimension to 42 vocab size 

    def forward(self, idx, targets=None):
        B, T = idx.shape # B = batch size, T = sequence length (block_size)
        tok_emb = self.token_embedding_table(idx) # (B, T, embedding_size)


        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, embedding_size)
        x = tok_emb + pos_emb # (B, T, embedding_size)
        x = self.blocks(x) # (B, T, embedding_size)
        x = self.ln_f(x) # (B, T, embedding_size)
        logits = self.lm_head(x) # (B, T, vocab_size) RAW prediction of the model

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss


    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:] # crop context to block size
            logits, _ = self(idx_cond) # get predictions
            logits = logits[:, -1, :] # focus only on last time step
            probs = F.softmax(logits, dim=-1) # get probabilities
            idx_next = torch.multinomial(probs, num_samples=1) # sample from the distribution
            idx = torch.cat((idx, idx_next), dim=1) # append sampled index to the running sequence

        return idx


model = TinyGPT()

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
for step in range(epochs):
    xb, yb = get_batch()
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if step % 300 == 0:
        print(f"step {step}: loss {loss.item():.4f}")