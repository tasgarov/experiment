import marimo

__generated_with = "0.18.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import polars

    return


@app.cell
def _():
    import torch
    from torch import nn
    import torch.nn.functional as F

    return F, nn, torch


@app.cell
def _(torch):
    x = torch.rand(4, 3, 5)
    x
    return (x,)


@app.cell
def _(x):
    x.transpose(1, 2)
    return


@app.cell
def _(torch, x):
    raw_weights = torch.bmm(x, x.transpose(1, 2))
    raw_weights
    return (raw_weights,)


@app.cell
def _(F, raw_weights):
    weights = F.softmax(raw_weights, dim=2)
    weights
    return (weights,)


@app.cell
def _(torch, weights, x):
    y = torch.bmm(weights, x)
    y
    return


@app.cell
def _(torch):
    Wq, Wk, Wv = torch.rand(5, 5), torch.rand(5, 5), torch.rand(5, 5)
    Wq
    return Wk, Wq, Wv


@app.cell
def _(Wk, Wq, Wv, x):
    q, k, v = x @ Wq, x @ Wk, x @ Wv
    q, k, v
    return k, q, v


@app.cell
def _(k, q, torch):
    W_raw = torch.bmm(q, k.transpose(1, 2)) / 5**0.5
    W_raw
    return (W_raw,)


@app.cell
def _(F, W_raw):
    W = F.softmax(W_raw, dim=2)
    W
    return (W,)


@app.cell
def _(W, torch, v):
    y2 = torch.bmm(W, v)
    y2
    return


@app.cell
def _(F, nn, torch):
    class SelfAttention(nn.Module):
        def __init__(self, k, head=4, mask=False):
            super().__init__()
            self.head = head
            self.k = k

            self.to_keys = nn.Linear(k, k, bias=False)
            self.to_queries = nn.Linear(k, k, bias=False)
            self.to_values = nn.Linear(k, k, bias=False)

            self.unifyheads = nn.Linear(k, k)

        def forward(self, x):
            b, t, embed_dim = x.size()
            h = self.head
            s = embed_dim // h

            q = self.to_queries(x).view(b, t, h, s)
            k = self.to_keys(x).view(b, t, h, s)
            v = self.to_values(x).view(b, t, h, s)

            k = k.transpose(1, 2).contiguous().view(b * h, t, s)
            q = q.transpose(1, 2).contiguous().view(b * h, t, s)
            v = v.transpose(1, 2).contiguous().view(b * h, t, s)

            W = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=2)
            y = torch.bmm(W, v).view(b, h, t, s)

            y = y.transpose(1, 2).contiguous().view(b, t, s * h)
            y = self.unifyheads(y)
            return y

    return (SelfAttention,)


@app.cell
def _(SelfAttention, nn):
    class TransformerBlock(nn.Module):
        def __init__(self, k, heads):
            super().__init__()

            self.attention = SelfAttention(k, heads=heads)

            self.norm1 = nn.LayerNorm(k)
            self.norm2 = nn.LayerNorm(k)

            self.ff = nn.Sequential(nn.Linear(k, 4 * k), nn.ReLU(), nn.Linear(4 * k, k))

        def forward(self, x):
            attention_layer = self.attention(x)
            x = self.norm1(attention_layer + x)

            ff_layer = self.ff(x)
            return self.norm2(ff_layer + x)

    return


app._unparsable_cell(
    r"""
    class Transformer(nn.Module):
        def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
            super().__init__()

            self.num_tokens = num_tokens
            self.token_emb = nn.Embedding(num_tokens, k)
            self.pos_emb = nn.Embedding(seq_length, k)

            tblocks = []
            for i in range(depth):
                tblocks.append(TransformerBlock(k=k, heads=heads))
            self.tblocks = nn.Sequential(*tblocks)

            self.toprobs = nn.Linear(k, num_classes)

        def forward(self, x):
            \"\"\"
            :param x: A (b, t) tensor of integer values representing
                      words (in some predetermined vocabulary).
            :return: A (b, c) tensor of log-probabilities over the
                     classes (where c is the nr. of classes).
            \"\"\"
    		# generate token embeddings
            tokens = self.token_emb(x)
            b, t, k = tokens.size()

    		# generate position embeddings
    		positions = torch.arange(t)
            positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

            x = tokens + positions
            x = self.tblocks(x)

            x = self.toprobs(x.mean(dim=1))
            return F.log_softmax(x, dim=1)
    """,
    name="_",
)


if __name__ == "__main__":
    app.run()
