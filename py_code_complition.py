import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.function
def loc(ds) -> int:
    return sum(len(c.split("\n")) for c in ds)


@app.function
def ds2string(ds) -> str:
    return "\n\n\n".join(c for c in ds)


@app.function
def get_unique_chars(sample_str: str) -> str:
    return ''.join(sorted(set(c for c in sample_str)))


@app.function
def create_vocabualary(unique_chars: str) -> dict:
    return {v: unique_chars.index(v) for v in unique_chars}


@app.cell
def _(torch):
    def get_batch(tensor_data: torch.Tensor, batch_size: int, block_size: int) -> torch.Tensor:
        ix = torch.randint(len(tensor_data) - block_size, (batch_size,))
        x = torch.stack([tensor_data[i:i+block_size] for i in ix])
        y = torch.stack([tensor_data[i+1:i+block_size+1] for i in ix])
        return x, y
    return (get_batch,)


@app.class_definition
class EncodeDecode:
    def __init__(self, vocab_mapping: dict) -> None:
        self.vocab_mapping = vocab_mapping
        self.reverse_vocab_mapping = {v: k for k, v in self.vocab_mapping.items()}

    def encode(self, raw_input: str) -> list[int]: 
        return [self.vocab_mapping[r] for r in raw_input]

    def decode(self, encoded: list[int]) -> str: 
        return "".join(self.reverse_vocab_mapping[e] for e in encoded)


@app.cell
def _(vocab_mapping):
    ende_code = EncodeDecode(vocab_mapping=vocab_mapping)
    return (ende_code,)


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return


@app.cell
def _():
    import torch
    from torch import nn
    import torch.nn.functional as F
    return F, nn, torch


@app.cell
def _():
    from datasets import load_dataset
    return (load_dataset,)


@app.cell
def _(load_dataset):
    ds = load_dataset("mbpp")
    return (ds,)


@app.cell
def _(ds):
    # there is no mistake here, test had more sample
    train_ds, test_ds, val_ds = ds["test"]["code"], ds["train"]["code"], ds["validation"]["code"]
    train_str, test_str, val_str = ds2string(train_ds), ds2string(test_ds), ds2string(val_ds)
    train_unique_chars, test_unique_chars, val_unique_chars = get_unique_chars(train_str), get_unique_chars(test_str), get_unique_chars(val_str)
    return (
        test_ds,
        test_str,
        test_unique_chars,
        train_ds,
        train_str,
        train_unique_chars,
        val_ds,
        val_str,
        val_unique_chars,
    )


@app.cell
def _(train_unique_chars):
    vocab_mapping = create_vocabualary(train_unique_chars)
    return (vocab_mapping,)


@app.cell
def _(test_ds, train_ds, val_ds):
    len(train_ds), len(test_ds), len(val_ds)
    return


@app.cell
def _(test_ds, train_ds, val_ds):
    loc(train_ds), loc(test_ds), loc(val_ds)
    return


@app.cell
def _(train_ds):
    print(train_ds[99])
    return


@app.cell
def _(test_unique_chars, train_unique_chars, val_unique_chars):
    (train_unique_chars, len(train_unique_chars)), (test_unique_chars, len(test_unique_chars)), (val_unique_chars, len(val_unique_chars))
    return


@app.cell
def _(ende_code, test_str, torch, train_str, val_str):
    train_tensor = torch.tensor(ende_code.encode(train_str))
    test_tensor = torch.tensor(ende_code.encode(test_str))
    val_tensor = torch.tensor(ende_code.encode(val_str))
    print(train_tensor[:10])
    return (train_tensor,)


@app.cell
def _(get_batch, train_tensor):
    x, y = get_batch(tensor_data=train_tensor, batch_size=4, block_size=8)
    x
    return x, y


@app.cell
def _(F, nn, torch):
    class BigramLanguageModel(nn.Module):

        def __init__(self, vocab_size, block_size, n_embd):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.pos_embedding = nn.Linear(block_size, n_embd)
            self.lm_head = nn.Linear(n_embd, vocab_size)
            self.block_size = block_size

        def forward(self, idx, targets=None):
            token_embd = self.token_embedding_table(idx) # (B,T,C)
            pos_embd = self.pos_embedding(torch.arange(self.block_size, dtype=torch.float))
            x = token_embd + pos_embd
            logits = self.lm_head(x)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                logits, loss = self(idx)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
    return (BigramLanguageModel,)


@app.cell
def _(BigramLanguageModel, vocab_mapping):
    bi_model = BigramLanguageModel(vocab_size=len(vocab_mapping), block_size=32, n_embd=64)
    bi_model
    return (bi_model,)


@app.cell
def _(bi_model, x, y):
    logits, loss = bi_model(x, y)
    print(logits)
    print(loss)
    return


@app.cell
def _(bi_model, ende_code, torch):
    print(ende_code.decode(bi_model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
    return


@app.cell
def _(get_batch, torch):
    def train(model, train_tensor, block_size: int, batch_size: int, epochs: int):
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        batch_size = batch_size
        for steps in range(epochs):
    
            xb, yb = get_batch(tensor_data=train_tensor, batch_size=batch_size, block_size=block_size)
    
            train_logits, train_loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            train_loss.backward(retain_graph=True)
            optimizer.step()
        return train_loss
    return (train,)


@app.cell
def _(bi_model, train, train_tensor):
    train_loss = train(model=bi_model, train_tensor=train_tensor, block_size=32, batch_size=64, epochs=1000)
    print(train_loss.item())
    return


@app.cell
def _(bi_model, ende_code, torch):
    print(ende_code.decode(bi_model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
    return


@app.cell
def _(F, nn, torch):
    class AttentionHead(nn.Module):
        def __init__(self, head_size, n_embd, block_size):
            super().__init__()
            self.to_key = nn.Linear(n_embd, head_size, bias=False)
            self.to_query = nn.Linear(n_embd, head_size, bias=False)
            self.to_value = nn.Linear(n_embd, head_size, bias=False)
        
            self.head_size = head_size
            self.tril = torch.tril(torch.ones(block_size, block_size))

        def forward(self, x):
            B, T, C = x.shape
            k = self.to_key(x)
            q = self.to_query(x)
            v = self.to_value(x)

            W = q @ k.transpose(-2, -1) / self.head_size ** 0.5
            W = W.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
            W = F.softmax(W, dim=-1)

            y = W @ v
            return y

    return (AttentionHead,)


@app.cell
def _(AttentionHead, F, nn, torch):
    class BigramAttentionLanguageModel(nn.Module):

        def __init__(self, vocab_size, head_size, block_size, n_embd):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.pos_embedding = nn.Linear(block_size, n_embd)
            self.sa_head = AttentionHead(head_size=head_size, n_embd=n_embd, block_size=block_size)
            self.lm_head = nn.Linear(n_embd, vocab_size)

            self.block_size = block_size

        def forward(self, idx, targets=None):
            token_embd = self.token_embedding_table(idx) # (B,T,C)
            pos_embd = self.pos_embedding(torch.arange(self.block_size, dtype=torch.float))
            x = token_embd + pos_embd
            x = self.sa_head(x)
            logits = self.lm_head(x)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
    return (BigramAttentionLanguageModel,)


@app.cell
def _(BigramAttentionLanguageModel, vocab_mapping):
    bi_attention_model = BigramAttentionLanguageModel(vocab_size=len(vocab_mapping), head_size=64, block_size=32, n_embd=64)
    bi_attention_model
    return (bi_attention_model,)


@app.cell
def _(bi_attention_model, train, train_tensor):
    train_loss2 = train(model=bi_attention_model, train_tensor=train_tensor, block_size=32, batch_size=64, epochs=100)
    print(train_loss2.item())
    return


@app.cell
def _(bi_attention_model, ende_code, torch):
    print(ende_code.decode(bi_attention_model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
    return


@app.cell
def _(AttentionHead, F, block_size, nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(self, num_heads, head_size):

            super().__init__()
            self.heads = [nn.ModuleList((AttentionHead(block_size=block_size, head_size=head_size))) for _ in range(num_heads)]

        def forward(self, x):
            return torch.cat([h(x) for h in self.heads], dim=-1)

    class BigramMultiHeadAttentionLanguageModel(nn.Module):

        def __init__(self, vocab_size, head_size, block_size, n_embd):
            super().__init__()
            self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
            self.pos_embedding = nn.Linear(block_size, n_embd)
            self.sa_head = MultiHeadAttention()
            self.lm_head = nn.Linear(n_embd, vocab_size)

            self.block_size = block_size

        def forward(self, idx, targets=None):
            token_embd = self.token_embedding_table(idx) # (B,T,C)
            pos_embd = self.pos_embedding(torch.arange(self.block_size, dtype=torch.float))
            x = token_embd + pos_embd
            x = self.sa_head(x)
            logits = self.lm_head(x)

            if targets is None:
                loss = None
            else:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)

            return logits, loss

        def generate(self, idx, max_new_tokens):
            # idx is (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                idx_cond = idx[:, -self.block_size:]
                logits, loss = self(idx_cond)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, idx_next), dim=1)
            return idx
    return


if __name__ == "__main__":
    app.run()
