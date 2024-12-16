"""
PyTorch implementation for ViT.

For more details, see:
[1] Dosovitskiy et.al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.
    http://arxiv.org/abs/2010.11929
"""

import torch
from torch import nn, Tensor


def _pair(x: int | tuple[int, int]) -> tuple[int]:
    return x if isinstance(x, tuple) else (x, x)


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        num_channels: int,
        num_classes: int,
        d_model: int,
        d_hidden: int,
        num_blocks: int,
        num_head: int,
        drop_prob: float = 0.0,
        emb_drop_prob: float = 0.0,
    ) -> None:
        super(ViT, self).__init__()
        image_height, image_width = _pair(image_size)
        patch_height, patch_width = _pair(patch_size)
        if image_height % patch_height != 0 or image_width % patch_width != 0:
            raise RuntimeError(f"Image dimensions must be divisable by the patch size.")
        num_patches = (image_height * image_width) // (patch_height * patch_width)

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        # embedding layers
        self.patch_embedding = PatchEmbedding(num_channels, d_model, patch_size)
        self.pos_embbeding = nn.Parameter(torch.randn(1, num_patches + 1, d_model))
        self.dropout = nn.Dropout(emb_drop_prob)
        # transformer encoder
        self.enc = TransformerEncoder(
            d_model, d_hidden, num_blocks, num_head, drop_prob
        )
        # classifier head
        self.mlp_head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor):
        batch_size = x.size()[0]
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
        out = self.patch_embedding(x)
        batch_size, num_patchs, _ = out.size()
        out = torch.cat([cls_tokens, out], dim=1)
        out += self.pos_embbeding[:, : (num_patchs + 1)]
        out = self.dropout(out)
        out = self.enc(out)[:, 0]

        # Perform classification
        out = self.mlp_head(out)
        return out


class PatchEmbedding(nn.Module):
    """Patch Embedding Layer in ViT"""

    def __init__(
        self,
        num_channels: int,
        d_model: int,
        patch_size: tuple[int, int],
    ) -> None:
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(
            num_channels, d_model, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor):
        out = self.proj(x)
        out = torch.flatten(out, start_dim=2).transpose(1, 2)
        return out


class TransformerEncoder(nn.Module):
    """Transformer Encoder in ViT"""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        num_blocks: int,
        num_head: int,
        drop_prob: float = 0.0,
    ) -> None:
        """Initialize a Transformer Encoder

        Args:
            d_model (int): dimensionality of model
            d_hidden (int): dimensionality of hidden layer in MLP
            num_blocks (int): number of encoder blocks
            num_head (int): number of heads in multi-head attention
            drop_prob (float, optional): dropout ratio. Defaults to 0.0.
        """
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(d_model, d_hidden, num_head, drop_prob)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class EncoderBlock(nn.Module):
    """ViT Encoder Block"""

    def __init__(
        self, d_model: int, d_hidden: int, num_head: int, drop_prob: float = 0.0
    ) -> None:
        super(EncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_head)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_hidden, drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x: Tensor):
        shortcut = x
        out = self.ln1(x)
        out = self.attention(out, out, out)
        out = self.dropout1(out)
        out += shortcut

        shortcut = out
        out = self.ln2(out)
        out = self.mlp(out)
        out = self.dropout2(out)
        out += shortcut
        return out


class MLP(nn.Module):
    """MultiLayer Perceptron with one hidden layer in ViT"""

    def __init__(self, d_model: int, d_hidden: int, drop_prob: float) -> None:
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x: Tensor):
        out = self.fc1(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self) -> None:
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax()

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        d_k = k.size()[3]
        kt = k.transpose(2, 3)
        # 1. Scale and dot product
        score = torch.matmul(q, kt) / d_k**0.5
        # 2. Apply softmax to get the attention score
        score = self.softmax(score)
        # 3. Compute weighted sum
        values = torch.matmul(score, v)
        return values


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_head: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.attention = ScaledDotProductAttention()
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.proj_o = nn.Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor):
        # 1. Apply linear projection
        q, k, v = self.proj_q(q), self.proj_k(k), self.proj_v(v)
        # 2. Split tensor by number of heads
        q, k, v = self._split(q), self._split(k), self._split(v)
        # 3. Do scaled dot-product attention
        out = self.attention(q, k, v)
        out = self._concat(out)
        out = self.proj_o(out)
        return out

    def _split(self, x: Tensor) -> Tensor:
        """Split tensor by number of heads

        Args:
            x (Tensor): input tensor of shape (batch_size, seq_length, d_model)

        Returns:
            Tensor: output tensor after split, of shape (batch_size, num_head, seq_length, d_head)
        """
        batch_size, seq_length, d_model = x.size()
        d_head = d_model // self.num_head
        return x.view(batch_size, seq_length, self.num_head, d_head).transpose(1, 2)

    def _concat(self, x: Tensor) -> Tensor:
        """Inverse function of self._split

        Args:
            x (Tensor): input tensor of shape (batch_size, num_head, seq_length, d_head)

        Returns:
            Tensor: output tensor after concatenation, of shape (batch_size, seq_length, d_model)
        """
        batch_size, num_head, seq_length, d_head = x.size()
        d_model = num_head * d_head
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, d_model)


def vit_extra_small(image_size: int | tuple[int, int], num_channels: int = 3) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=4,
        num_channels=num_channels,
        d_model=192,
        d_hidden=768,
        num_blocks=6,
        num_head=3,
        drop_prob=0.1,
        emb_drop_prob=0.1,
    )


def vit_small(image_size: int | tuple[int, int], num_channels: int = 3) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=4,
        num_channels=num_channels,
        d_model=384,
        d_hidden=1536,
        num_blocks=8,
        num_head=6,
        drop_prob=0.1,
        emb_drop_prob=0.1,
    )


def vit_base(image_size: int | tuple[int, int], num_channels: int = 3) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=4,
        num_channels=num_channels,
        d_model=768,
        d_hidden=3072,
        num_blocks=12,
        num_head=12,
        drop_prob=0.1,
        emb_drop_prob=0.1,
    )


def vit_large(image_size: int | tuple[int, int], num_channels: int = 3) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=4,
        num_channels=num_channels,
        d_model=1024,
        d_hidden=4096,
        num_blocks=24,
        num_head=16,
        drop_prob=0.1,
        emb_drop_prob=0.1,
    )


def vit_huge(image_size: int | tuple[int, int], num_channels: int = 3) -> ViT:
    return ViT(
        image_size=image_size,
        patch_size=4,
        num_channels=num_channels,
        d_model=1280,
        d_hidden=5120,
        num_blocks=32,
        num_head=16,
        drop_prob=0.1,
        emb_drop_prob=0.1,
    )
