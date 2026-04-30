# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from gr00t.model.action_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from .cross_attention_dit import DiT, SelfAttentionTransformer


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x
    
class MultiScaleMaskAwareConv(nn.Module):
    def __init__(self, input_dim=2048, output_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Multi-scale convolution branches
        self.conv_branches = nn.ModuleList([
            # Branch 1: Small kernel
            nn.Sequential(
                nn.Conv1d(input_dim, 512, kernel_size=3, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            ),
            # Branch 2: Medium kernel
            nn.Sequential(
                nn.Conv1d(input_dim, 512, kernel_size=5, padding=2),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 256, kernel_size=5, padding=2),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            ),
            # Branch 3: Large kernel
            nn.Sequential(
                nn.Conv1d(input_dim, 512, kernel_size=7, padding=3),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Conv1d(512, 256, kernel_size=7, padding=3),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            )
        ])
        
        # Feature fusion and MLP
        self.mlp = nn.Sequential(
            nn.Linear(768, 512),  # 3 branches * 256 = 768
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        
        # Initialize weights
        # self.apply(initialize_weights)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, features]
            mask: Attention mask, shape [batch_size, seq_len], 1 for valid positions, 0 for padding
        """
        # Transpose to shape expected by convolution layers
        x = x.transpose(1, 2)  # Shape becomes [batch_size, features, seq_len]
        
        # Prepare mask
        if mask is not None:
            mask = mask.unsqueeze(1).float()  # Shape becomes [batch_size, 1, seq_len]
        
        # Pass through multi-scale convolution branches
        branch_outputs = []
        for branch in self.conv_branches:
            branch_out = branch(x)  # Each branch output shape: [batch_size, 256, seq_len]
            
            # Apply mask-aware pooling - fix shape issue
            if mask is not None:
                masked_branch_out = branch_out * mask
                # Count valid elements
                valid_count = mask.sum(dim=2)  # Shape becomes [batch_size, 1]
                # Compute mask-aware average pooling
                pooled = masked_branch_out.sum(dim=2)  # Shape becomes [batch_size, 256]
                pooled = pooled / valid_count.clamp_min(1e-8)  # Shape remains [batch_size, 256]
            else:
                # If no mask, use regular global average pooling
                pooled = torch.mean(branch_out, dim=2)  # Shape becomes [batch_size, 256]
                
            branch_outputs.append(pooled)
        
        # Merge branch outputs
        x = torch.cat(branch_outputs, dim=1)  # Shape becomes [batch_size, 768]
        
        # Pass through MLP
        output = self.mlp(x)  # Shape becomes [batch_size, output_dim]
        
        return output
    
class MaskAwareAttentionPooling(nn.Module):
    def __init__(self, input_dim=2048, output_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1D convolution for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 1, kernel_size=1),
        )
        
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights
        # self.apply(initialize_weights)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, features]
            mask: Attention mask, shape [batch_size, seq_len], 1 for valid positions, 0 for padding
        """
        # Transpose to shape expected by convolution layers
        x = x.transpose(1, 2)  # Shape becomes [batch_size, features, seq_len]
        
        # Pass through convolution layers
        conv_out = self.conv_layers(x)  # Shape becomes [batch_size, 512, seq_len]
        
        # Compute attention scores
        attention_scores = self.attention(conv_out)  # Shape becomes [batch_size, 1, seq_len]
        
        # Apply mask to attention scores
        if mask is not None:
            mask = mask.unsqueeze(1)  # Shape becomes [batch_size, 1, seq_len]
            # Set attention scores for padding positions to negative infinity
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=2)  # Shape becomes [batch_size, 1, seq_len]
        
        # Apply attention weights
        weighted_features = conv_out * attention_weights  # Shape becomes [batch_size, 512, seq_len]
        
        # Global sum pooling - fix shape issue
        pooled = weighted_features.sum(dim=2)  # Shape becomes [batch_size, 512]
        
        # Pass through MLP
        output = self.mlp(pooled)  # Shape becomes [batch_size, output_dim]
        
        return output

class Conv1DAttentionProcessor(nn.Module):
    def __init__(self, input_dim=2048, output_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1D convolution for feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )
        
        # Attention pooling
        self.attention_pool = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(256, 1, kernel_size=1),
            nn.Sigmoid()  # Attention weights, range [0,1]
        )
        
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x, mask=None):
        # Input shape: [batch_size, seq_len, features] -> [2, 290, 2048]
        # Transpose to shape expected by convolution layers: [batch_size, features, seq_len]
        x = x.transpose(1, 2)  # Shape becomes [2, 2048, 290]
        
        # Pass through convolution layers
        conv_out = self.conv_layers(x)  # Shape becomes [2, 512, 290]
        
        if mask is not None:
            # Convert mask to same shape as convolution output
            mask = mask.unsqueeze(1).float()  # Shape becomes [batch_size, 1, seq_len]
            
            # Apply mask to convolution output
            masked_conv_out = conv_out * mask
            # Compute attention weights
            attention_weights = self.attention_pool(masked_conv_out)  # Shape becomes [2, 1, 290]
            # Apply attention weights
            weighted_features = masked_conv_out * attention_weights  # Shape becomes [2, 512, 290]
            
            # Count valid elements
            valid_count = mask.sum(dim=2)  # Shape becomes [batch_size, 1]
            
            # Compute mask-aware average pooling
            pooled = weighted_features.sum(dim=2) / valid_count.clamp_min(1e-8)  # Shape becomes [batch_size, 512]
        else:
            attention_weights = self.attention_pool(conv_out)  # Shape becomes [2, 1, 290]
            # Apply attention weights
            weighted_features = conv_out * attention_weights  # Shape becomes [2, 512, 290]
            # If no mask, use regular global average pooling
            pooled = torch.mean(weighted_features, dim=2)  # Shape becomes [batch_size, 512]
        
        
        
        # Pass through MLP
        output = self.mlp(pooled)  # Shape becomes [2, output_dim]
        
        return output



class MaskAwareConv1DProcessor(nn.Module):
    def __init__(self, input_dim=2048, hidden_dims=[1024, 512], output_dim=6):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 1D convolution layers for sequence processing
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        
        # MLP part
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor, shape [batch_size, seq_len, features]
            mask: Attention mask, shape [batch_size, seq_len], 1 for valid positions, 0 for padding
        """
        # Input shape: [batch_size, seq_len, features] -> [2, 290, 2048]
        # Transpose to shape expected by convolution layers: [batch_size, features, seq_len]
        x = x.transpose(1, 2)  # Shape becomes [batch_size, features, seq_len]
        
        # Pass through convolution layers
        conv_out = self.conv_layers(x)  # Shape becomes [batch_size, 256, seq_len]
        
        # Apply mask-aware global pooling
        if mask is not None:
            # Convert mask to same shape as convolution output
            mask = mask.unsqueeze(1).float()  # Shape becomes [batch_size, 1, seq_len]
            
            # Apply mask to convolution output
            masked_conv_out = conv_out * mask
            
            # Count valid elements
            valid_count = mask.sum(dim=2)  # Shape becomes [batch_size, 1, 1]
            
            # Compute mask-aware average pooling
            pooled = masked_conv_out.sum(dim=2) / valid_count.clamp_min(1e-8)  # Shape becomes [batch_size, 256]
        else:
            # If no mask, use regular global average pooling
            pooled = torch.mean(conv_out, dim=2)  # Shape becomes [batch_size, 256]
        
        # Pass through MLP
        output = self.mlp(pooled)  # Shape becomes [batch_size, output_dim]
        
        return output

class TargetModel(nn.Module):
    def __init__(self, target_pos_dim=6):
        super(TargetModel, self).__init__()
        self.target_tokens = nn.Embedding(1, 512)
        nn.init.normal_(self.target_tokens.weight, mean=0.0, std=0.02)

        self.target_norm1 = nn.LayerNorm(512, elementwise_affine=True, eps=1e-5)
        self.target_norm2 = nn.LayerNorm(512, elementwise_affine=True, eps=1e-5)
        self.target_norm3 = nn.LayerNorm(512, elementwise_affine=True, eps=1e-5)
        
        self.target_model = nn.MultiheadAttention(512, 8, batch_first=True)
        
        self.target_decoder = nn.Linear(512, target_pos_dim)
        self.target_encoder = nn.Linear(2048, 512)

    def forward(self, vl_embs, vl_attn_mask):
        """
        Forward pass for the target model.

        Args:
            vl_embs (Tensor): Visual embeddings of shape (batch_size, seq_len, 2048).
            vl_attn_mask (Tensor): Attention mask for the visual embeddings.

        Returns:
            Tensor: Predicted target positions.
        """
        # Expand target tokens to match the batch size
        target_tokens = self.target_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)

        # Encode visual embeddings
        vl_embs_flat = vl_embs.reshape(-1, vl_embs.shape[-1])
        vl_embs_mini = self.target_encoder(vl_embs_flat)
        vl_embs_mini = vl_embs_mini.reshape(vl_embs.shape[0], vl_embs.shape[1], -1)
        
        # Normalize tokens and visual embeddings
        target_tokens = self.target_norm1(target_tokens)
        vl_embs_mini = self.target_norm2(vl_embs_mini)
        
        # Multi-head attention
        target_output, _ = self.target_model(
            target_tokens,
            vl_embs_mini,
            vl_embs_mini,
            key_padding_mask=~vl_attn_mask.to(torch.bool),
        )
        
        # Reshape and normalize output
        target_output = target_output.reshape(target_output.shape[0], -1)
        target_output = self.target_norm3(target_output)
        
        # Decode to target positions
        target_pred = self.target_decoder(target_output)
        
        return target_pred


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    model_dtype: str = field(default="float32", metadata={"help": "Model data type."})
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )
    backbone_embedding_dim: int = field(
        default=1536, metadata={"help": "Backbone embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    target_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for target loss. Set to 0 to disable target loss for ablation."},
    )

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FlowmatchingActionHead(nn.Module):
    config_class = FlowmatchingActionHeadConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: FlowmatchingActionHeadConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        self.model = DiT(**config.diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)
        
        
        self.target_model = Conv1DAttentionProcessor(
            input_dim=config.backbone_embedding_dim,
            output_dim=getattr(config, "max_target_pos_dim", 6),
        )
        

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.vl_self_attention = (
            SelfAttentionTransformer(**config.vl_self_attention_cfg)
            if config.use_vlln
            else nn.Identity()
        )
        
        
        # Conv1DAttentionProcessor(2048,self.target_pos_dim)
        # 
        
        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config
        self.set_trainable_parameters(config.tune_projector, config.tune_diffusion_model)

    def set_trainable_parameters(self, tune_projector: bool, tune_diffusion_model: bool):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        print(f"Tune action head projector: {self.tune_projector}")
        print(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        # Check if any parameters are still trainable. If not, print a warning.
        if not tune_projector and not tune_diffusion_model:
            for name, p in self.named_parameters():
                if p.requires_grad:
                    print(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            print("Warning: No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        backbone_output = self.process_backbone_output(backbone_output)

        if self.config.expand_batch is not None:
            for k, v in backbone_output.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                backbone_output[k] = expanded

            for k, v in action_input.items():
                ndim = len(v.shape)
                factors = [self.config.expand_batch]
                while len(factors) < ndim:
                    factors.append(1)
                factors = tuple(factors)
                expanded = v.repeat(*factors)
                action_input[k] = expanded

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        device = vl_embs.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

        vl_attn_mask = backbone_output.backbone_attention_mask

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1] :]
        
        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        loss_action = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss_action = loss_action.sum() / action_mask.sum()
        
        # Target loss (conditional on target_loss_weight > 0)
        target_loss_weight = getattr(self.config, 'target_loss_weight', 1.0)
        if target_loss_weight > 0:
            target_pred = self.target_model(vl_embs, vl_attn_mask)
            target_pos_mask = action_input.target_pos_mask.reshape(target_pred.shape[0], -1)
            target_pos = action_input.target_pos.reshape(target_pred.shape[0], -1)
            loss_target = F.mse_loss(target_pred, target_pos, reduction="none") * target_pos_mask
            loss_target = loss_target.sum() / target_pos_mask.sum()
            loss = loss_action + target_loss_weight * loss_target
            print('loss_action', loss_action)
            print('loss_target', loss_target)
        else:
            loss = loss_action
            print('loss_action', loss_action)
            print('target_loss disabled (target_loss_weight=0)')
        
        output_dict = {
            "loss": loss
        }
        return BatchFeature(data=output_dict)

    @torch.no_grad()
    def get_action(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embs = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        
        # Target prediction (conditional on target_loss_weight > 0)
        target_loss_weight = getattr(self.config, 'target_loss_weight', 1.0)
        if target_loss_weight > 0:
            target_pred = self.target_model(vl_embs)
        else:
            target_pred = None

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output, embodiment_id)

            pred_velocity = pred[:, -self.action_horizon :]

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        result = {"action_pred": actions}
        if target_pred is not None:
            result["target_pred"] = target_pred
        return BatchFeature(data=result)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
