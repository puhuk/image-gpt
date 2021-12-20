import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

class GPT2(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=12, num_positions=1024, num_vocab=512, num_classes=10, warmup_steps=500, steps=10_000):

        super(GPT2, self).__init__()

        self.embed_dim = embed_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.warmup_steps = warmup_steps
        self.steps = steps

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(TransformerBlock(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def configure_optimizers(self, config):
        optimizer = torch.optim.Adam(self.parameters(), lr=config.learning_rate)

        # no learning rate schedule for fine-tuning
        # if self.classify:
        #     return optimizer

        scheduler = {
            "scheduler": LambdaLR(
                optimizer, learning_rate_schedule(self.warmup_steps, self.steps)
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]


    def forward(self, x, classify=False):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        print(x.shape)
        length, batch = x.shape

        h = self.token_embeddings(x)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)

        logits = self.head(h)

        return logits

        # if not classify:
        #     # return logits
        #     return logits

        # h = torch.mean(h, dim=0)  # average pool over sequence
        # return classification logits and generative logits

        # clf_loss = self.criterion(self.clf_head(h), y)
        # gen_loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
        # loss = clf_loss + gen_loss
        # loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
        # return self.clf_head(h), logits
        # return logits, loss

def learning_rate_schedule(warmup_steps, total_steps):
    """Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps"""

    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn