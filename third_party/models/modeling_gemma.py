"""
Gemma-3 text-only decoder with Generative Adapter (Hyperformer) hooks.

Architecture matches HuggingFace's Gemma3 *exactly* so that pretrained weights
load without renaming.  Adapter modules are added as **extra** attributes on
`Gemma3DecoderLayer` and `Gemma3TextModel`, so they don't clash with the
pretrained state-dict.

Weight-name parity (matches `google/gemma-3-1b-it`):
    model.embed_tokens.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.self_attn.{q,k}_norm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight
    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.pre_feedforward_layernorm.weight
    model.layers.{i}.post_feedforward_layernorm.weight
    model.norm.weight
    lm_head.weight

Adapter loading pattern (two-step):
    1.  model = Gemma3ForConditionalGeneration.from_pretrained(path, config=config)
    2.  model._init_adapter_modules(adapter_config)   # wires adapters post-load
"""

# ────────────────────────────── stdlib ──────────────────────────────
import copy
from collections.abc import Callable
from typing import Optional, Tuple, Union

# ─────────────────────────────── torch ──────────────────────────────
import torch
from torch import nn

# ──────────────────────── transformers utils ────────────────────────
from transformers.utils import logging, auto_docstring
from transformers.utils.generic import check_model_inputs
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

# ──────────────────── transformers modelling ────────────────────────
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.activations import ACT2FN

# ────────────── HF Gemma-3 building blocks we reuse ────────────────
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3PreTrainedModel,
    Gemma3RotaryEmbedding,
    Gemma3TextScaledWordEmbedding,
    Gemma3CausalLMOutputWithPast,
    _bidirectional_window_overlay,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

# ───────────────── adapter controllers (hyperformer) ───────────────
from adapters import (
    AutoAdapterController,
    MetaAdapterConfig,
    TaskEmbeddingController,
    AdapterLayersHyperNetController,
    MetaLayersAdapterController,
    AdapterLayersOneHyperNetController,
)

# ───────────────────── local text config ───────────────────────────
from .config_gemma3 import Gemma3TextConfig

logger = logging.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1.  Gemma3RMSNorm  (matches HF exactly)
# ═══════════════════════════════════════════════════════════════════
class Gemma3RMSNorm(nn.Module):
    """RMSNorm with Gemma-style (1+w) scaling."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ═══════════════════════════════════════════════════════════════════
# 2.  Gemma3MLP  (same weight names as pretrained)
# ═══════════════════════════════════════════════════════════════════
class Gemma3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# ═══════════════════════════════════════════════════════════════════
# 3.  Gemma3Attention
#     q_proj, k_proj, v_proj, o_proj, q_norm, k_norm
#     → parameter names match pretrained exactly
# ═══════════════════════════════════════════════════════════════════
class Gemma3Attention(nn.Module):
    """Multi-headed attention with QK-norm (Gemma-3 style)."""

    def __init__(self, config: Gemma3TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = not getattr(config, "use_bidirectional_attention", False)

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.attn_logit_softcapping = getattr(config, "attn_logit_softcapping", None)
        self.sliding_window = config.sliding_window if self.is_sliding else None

        # QK norms (Gemma-3 specific)
        self.q_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # QK normalization
        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            softcap=self.attn_logit_softcapping,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ═══════════════════════════════════════════════════════════════════
# 4.  Gemma3DecoderLayer
#     Standard HF parameter names:
#       self_attn, mlp, input_layernorm, post_attention_layernorm,
#       pre_feedforward_layernorm, post_feedforward_layernorm
#     Adapter modules are EXTRA attributes (not in pretrained).
# ═══════════════════════════════════════════════════════════════════
class Gemma3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Gemma3TextConfig, layer_idx: int, adapter_config=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        # ── Standard HF Gemma-3 sub-modules (weight names match pretrained) ──
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # ── Adapter state flags ──
        self.train_adapters = False
        self.unique_hyper_net = False
        self.train_adapters_blocks = False

        # Optionally wire adapters at construction time
        if getattr(config, "train_adapters", False) and adapter_config is not None:
            self._init_adapter_modules(adapter_config)

    def _init_adapter_modules(self, adapter_config):
        """
        Attach adapter controllers to this decoder layer.
        Safe to call after from_pretrained(); does not touch pretrained weights.
        """
        if adapter_config is None:
            return

        self.train_adapters = True
        self.unique_hyper_net = isinstance(adapter_config, MetaAdapterConfig) and (
            getattr(adapter_config, "unique_hyper_net", False)
            or getattr(adapter_config, "efficient_unique_hyper_net", False)
        )
        self.train_adapters_blocks = (
            getattr(adapter_config, "train_adapters_blocks", False)
            and not self.unique_hyper_net
        )

        if self.train_adapters_blocks:
            # Per-task static adapters for attention + FF
            self.attn_adapter_controller = AutoAdapterController.get(adapter_config)
            self.ff_adapter_controller = AutoAdapterController.get(adapter_config)
            self.is_meta_adapter = isinstance(adapter_config, MetaAdapterConfig)
        elif self.unique_hyper_net:
            # Hyper-net generated adapters for attention + FF
            self.attn_layer_hyper_net = MetaLayersAdapterController(adapter_config)
            self.ff_layer_hyper_net = MetaLayersAdapterController(adapter_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings_global: Tuple[torch.Tensor, torch.Tensor],
        position_embeddings_local: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        # adapter signals
        task=None,
        task_embedding=None,
        gemma_adapters=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, ...]:
        # Strip adapter kwargs before passing to attention
        for k in ("task", "task_embedding", "gemma_adapters"):
            kwargs.pop(k, None)

        # ── Self-attention block (sandwich norm) ──
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Select global vs local RoPE
        if self.self_attn.is_sliding:
            position_embeddings = position_embeddings_local
        else:
            position_embeddings = position_embeddings_global

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        # >>> ADAPTER: inject on attention output <<<
        if self.train_adapters:
            if self.train_adapters_blocks:
                key = task if not getattr(self, "is_meta_adapter", False) else task_embedding
                hidden_states = self.attn_adapter_controller(key, hidden_states)
            elif self.unique_hyper_net and gemma_adapters is not None:
                sa = getattr(gemma_adapters, "self_attention", None)
                if sa is not None:
                    hidden_states = self.attn_layer_hyper_net(hidden_states, sa)

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # ── Feed-forward block (sandwich norm) ──
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # >>> ADAPTER: inject on FF output <<<
        if self.train_adapters:
            if self.train_adapters_blocks:
                key = task if not getattr(self, "is_meta_adapter", False) else task_embedding
                hidden_states = self.ff_adapter_controller(key, hidden_states)
            elif self.unique_hyper_net and gemma_adapters is not None:
                ff = getattr(gemma_adapters, "feed_forward", None)
                if ff is not None:
                    hidden_states = self.ff_layer_hyper_net(hidden_states, ff)

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ═══════════════════════════════════════════════════════════════════
# 5.  Gemma3TextModel  (text-only decoder stack)
#     Weight names: embed_tokens, layers, norm, rotary_emb, rotary_emb_local
# ═══════════════════════════════════════════════════════════════════
class Gemma3TextModel(Gemma3PreTrainedModel):
    """
    Gemma-3 text-only decoder stack with adapter plumbing.
    Matches HF's Gemma3TextModel parameter names exactly.
    """

    config: Gemma3TextConfig
    input_modalities = "text"

    def __init__(
        self,
        config: Gemma3TextConfig,
        embed_tokens: Optional[nn.Embedding] = None,
        adapter_config=None,
    ):
        super().__init__(config)
        self.adapter_config = adapter_config
        self.gradient_checkpointing = False

        # ── Embeddings (name: embed_tokens) ──
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = (
            embed_tokens
            if embed_tokens is not None
            else Gemma3TextScaledWordEmbedding(
                config.vocab_size,
                config.hidden_size,
                self.padding_idx,
                embed_scale=config.hidden_size ** 0.5,
            )
        )

        # ── Decoder layers (name: layers) ──
        self.layers = nn.ModuleList(
            [
                Gemma3DecoderLayer(config, layer_idx=i, adapter_config=adapter_config)
                for i in range(config.num_hidden_layers)
            ]
        )

        # ── Final norm (name: norm) ──
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── Rotary embeddings ──
        # Global RoPE: uses config.rope_theta (default 1_000_000)
        self.rotary_emb = Gemma3RotaryEmbedding(config=config)

        # Local RoPE: uses config.rope_local_base_freq (default 10_000)
        local_config = copy.deepcopy(config)
        local_config.rope_theta = config.rope_local_base_freq
        local_config.rope_scaling = {"rope_type": "default"}
        self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_config)

        # ── Adapter stack-level state flags ──
        self.train_adapters = False
        self.unique_hyper_net = False
        self.efficient_unique_hyper_net = False

        if getattr(config, "train_adapters", False) and adapter_config is not None:
            self._init_adapter_modules(adapter_config)

        self.post_init()

    def _init_adapter_modules(self, adapter_config):
        """
        Wire up adapter sub-modules onto this text model and its decoder layers.
        Call AFTER from_pretrained() to attach adapters without disturbing
        pretrained weights.
        """
        if adapter_config is None:
            return

        self.adapter_config = adapter_config
        self.train_adapters = True
        self.config.train_adapters = True

        self.unique_hyper_net = isinstance(adapter_config, MetaAdapterConfig) and bool(
            getattr(adapter_config, "unique_hyper_net", False)
        )
        self.efficient_unique_hyper_net = isinstance(adapter_config, MetaAdapterConfig) and bool(
            getattr(adapter_config, "efficient_unique_hyper_net", False)
        )

        if self.unique_hyper_net:
            self.adapter_layers_hyper_net = AdapterLayersHyperNetController(
                adapter_config, self.config.num_hidden_layers
            )
        if self.efficient_unique_hyper_net:
            self.adapter_layers_hyper_net = AdapterLayersOneHyperNetController(
                adapter_config, self.config.num_hidden_layers
            )

        # Wire up per-layer adapter modules
        for layer in self.layers:
            layer._init_adapter_modules(adapter_config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.embed_tokens = new_embeddings

    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        # adapter hooks
        task: Optional[str] = None,
        task_embedding: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. "
                "Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None and not self.training:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen,
                past_seen + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # ── Attention masks (per layer-type) ──
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            sliding_mask_kwargs = dict(mask_kwargs)

            if getattr(self.config, "use_bidirectional_attention", False):
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(
                    self.config.sliding_window
                )

            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
            }

        # ── Rotary position embeddings (global + local) ──
        hidden_states = inputs_embeds
        position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

        # ── Decoder loop ──
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Generate per-layer adapter weights from stack-level hyper-net
            gemma_adapters = None
            if self.train_adapters and (self.unique_hyper_net or self.efficient_unique_hyper_net):
                gemma_adapters = self.adapter_layers_hyper_net(task_embedding, i)

            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                task=task,
                task_embedding=task_embedding,
                gemma_adapters=gemma_adapters,
                **kwargs,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# ═══════════════════════════════════════════════════════════════════
# 6.  Gemma3ForConditionalGeneration  (text-only CausalLM)
#     Weight names:  model.*, lm_head.weight
# ═══════════════════════════════════════════════════════════════════
class Gemma3ForConditionalGeneration(Gemma3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, adapter_config=None):
        super().__init__(config)
        self.adapter_config = adapter_config
        self.train_adapters = bool(getattr(config, "train_adapters", False))

        # Task embedding controller (only when adapters are active)
        self.task_embedding_controller = None
        if self.train_adapters and isinstance(adapter_config, MetaAdapterConfig):
            self.task_embedding_controller = TaskEmbeddingController(adapter_config)

        # Text config
        self.text_config = getattr(config, "text_config", config)

        # Text decoder stack
        self.model = Gemma3TextModel(self.text_config, adapter_config=adapter_config)

        # LM head
        self.lm_head = nn.Linear(self.text_config.hidden_size, self.text_config.vocab_size, bias=False)

        self.post_init()

    def _init_adapter_modules(self, adapter_config):
        """
        Two-step loading: call this AFTER from_pretrained() to attach adapters
        without disturbing pretrained weights.

        Usage:
            model = Gemma3ForConditionalGeneration.from_pretrained(path, config=config)
            model._init_adapter_modules(adapter_config)
        """
        if adapter_config is None:
            return

        self.adapter_config = adapter_config
        self.train_adapters = True
        self.config.train_adapters = True
        self.text_config.train_adapters = True

        # Task embedding controller
        if isinstance(adapter_config, MetaAdapterConfig):
            self.task_embedding_controller = TaskEmbeddingController(adapter_config)

        # Delegate to text model
        self.model._init_adapter_modules(adapter_config)

    # ── weight-tying helpers ──
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # adapter hooks
        task: Optional[str] = None,
        task_embedding: Optional[torch.Tensor] = None,
        **lm_kwargs,
    ) -> Union[tuple, Gemma3CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Compute task embedding from task name if not already provided
        if (
            task_embedding is None
            and self.train_adapters
            and isinstance(self.adapter_config, MetaAdapterConfig)
            and self.task_embedding_controller is not None
        ):
            task_embedding = self.task_embedding_controller(task)

        # Strip eval-specific keys the text model doesn't need
        for key in ("input_ids_eval", "attention_mask_eval", "labels_eval"):
            lm_kwargs.pop(key, None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            task=task,
            task_embedding=task_embedding,
            **lm_kwargs,
        )

        hidden_states = outputs[0]

        # Logit slicing
        if isinstance(logits_to_keep, int):
            sl = slice(None) if logits_to_keep == 0 else slice(-logits_to_keep, None)
        else:
            sl = logits_to_keep
        logits = self.lm_head(hidden_states[:, sl, :])

        # Loss computation
        loss = None
        if labels is not None:
            logits_f = logits.float()
            shift_logits = logits_f[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1]:].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask != 0].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            flat_logits = shift_logits.view(-1, self.text_config.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )