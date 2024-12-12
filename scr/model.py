import numpy as np
from typing import Optional, Tuple, Union
import torch
from torch import nn
from dataclasses import dataclass
from transformer import LayerNorm, TextTransformer, VisionTransformer


def create_model(args, model_cfg):
    if args.PEFT=='adapter_image':
        model = CLIPWithImageAdapter(args, model_cfg).to(device=args.device)

    elif args.PEFT=='adapter_text':
        model = CLIPWithTextAdapter(args, model_cfg).to(device=args.device)

    elif args.PEFT=='adapter_two':
        model = CLIPWithtwoAdapter(args, model_cfg).to(device=args.device)

    elif args.PEFT is None:
        model = CLIP(model_cfg).to(device=args.device)

    return model


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def _build_vision_tower(embed_dim: int, vision_cfg: CLIPVisionCfg):
    vision_cfg = CLIPVisionCfg(**vision_cfg)
    act_layer = nn.GELU
    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer =  LayerNorm

    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return visual



def _build_text_tower(embed_dim: int, text_cfg: CLIPTextCfg):
    text_cfg = CLIPTextCfg(**text_cfg)
    act_layer = nn.GELU
    norm_layer = LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return text


class Adapter(nn.Module):
    def __init__(self, c_in, reduction):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.fc(x)
        return x



class CLIP(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.embed_dim = model_cfg["embed_dim"]
        self.vision_cfg = model_cfg["vision_cfg"]
        self.text_cfg = model_cfg["text_cfg"]
        
        self.visual = _build_vision_tower(self.embed_dim, self.vision_cfg)
        text = _build_text_tower(self.embed_dim, self.text_cfg)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # scaling factor that adjusts the magnitude of the similarity between image and text embeddings
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        features = self.visual(image)
        return features

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        text = text.squeeze(1)
        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features, self.logit_scale.exp()


class CLIPWithImageAdapter(nn.Module):
    def __init__(self, args, model_cfg):
        super().__init__()
        self.embed_dim = model_cfg["embed_dim"]
        self.vision_cfg = model_cfg["vision_cfg"]
        self.text_cfg = model_cfg["text_cfg"]
        
        # Vision and Text Encoders
        self.visual = _build_vision_tower(self.embed_dim, self.vision_cfg)
        text = _build_text_tower(self.embed_dim, self.text_cfg)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # Adapter layer added to the image encoder output
        self.adapter_image = Adapter(self.embed_dim, reduction=args.reduction).to(torch.float32)  # Adjust dtype as per model dtype

        # Logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        features = self.visual(image)
        # Apply Adapter
        adapter_output = self.adapter_image(features)
        
        ratio = 0.9
        mixed_features = ratio * adapter_output + (1 - ratio) * features
        
        return mixed_features

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        text = text.squeeze(1)
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features, self.logit_scale.exp()


class CLIPWithTextAdapter(nn.Module):
    def __init__(self, args, model_cfg):
        super().__init__()
        self.embed_dim = model_cfg["embed_dim"]
        self.vision_cfg = model_cfg["vision_cfg"]
        self.text_cfg = model_cfg["text_cfg"]
        
        # Vision and Text Encoders
        self.visual = _build_vision_tower(self.embed_dim, self.vision_cfg)
        text = _build_text_tower(self.embed_dim, self.text_cfg)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # Adapter layer added to the text encoder output
        self.adapter_text = Adapter(self.embed_dim, reduction=args.reduction).to(torch.float32)  # Adjust dtype as per model dtype

        # Logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        features = self.visual(image)

        return features

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        text = text.squeeze(1)
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        # Apply Adapter
        adapter_output = self.adapter_text(x)
        
        ratio = 0.9
        mixed_features = ratio * adapter_output + (1 - ratio) * x

        return mixed_features

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features, self.logit_scale.exp()


class CLIPWithtwoAdapter(nn.Module):
    def __init__(self, args, model_cfg):
        super().__init__()
        self.embed_dim = model_cfg["embed_dim"]
        self.vision_cfg = model_cfg["vision_cfg"]
        self.text_cfg = model_cfg["text_cfg"]
        
        # Vision and Text Encoders
        self.visual = _build_vision_tower(self.embed_dim, self.vision_cfg)
        text = _build_text_tower(self.embed_dim, self.text_cfg)
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        # Adapter layer added to the image and text encoder output
        self.adapter_image = Adapter(self.embed_dim, reduction=args.reduction).to(torch.float32)  # Adjust dtype as per model dtype
        self.adapter_text = Adapter(self.embed_dim, reduction=args.reduction).to(torch.float32)  # Adjust dtype as per model dtype
        # Logit scale parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_image(self, image):
        features = self.visual(image)
        # print('visual feature shape: {}'.format(features.shape))
        # Apply Adapter
        adapter_output = self.adapter_image(features)
        
        ratio = 0.9
        mixed_features = ratio * adapter_output + (1 - ratio) * features
        
        return mixed_features

    def encode_text(self, text):
        cast_dtype = self.transformer.get_cast_dtype()

        text = text.squeeze(1)
        x = self.token_embedding(text).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        adapter_output = self.adapter_text(x)
        
        ratio = 0.9
        mixed_features = ratio * adapter_output + (1 - ratio) * x
        
        return mixed_features

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return image_features, text_features, self.logit_scale.exp()
