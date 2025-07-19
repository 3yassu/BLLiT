#BLLiT_model.py
from transformers import PreTrainedModel
@auto_docstring
class BllitPreTrainedModel(PreTrainedModel):
    config: BllitConfig
    #base_model_prefix = "bllit"
	print("Unimplemented")
    """
    supports_gradient_checkpointing = True
    _supports_attention_backend = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _no_split_modules = [
        "Blip2Attention",
        "Blip2QFormerMultiHeadAttention",
        "Blip2EncoderLayer",
        "Blip2TextEmbeddings",
        "T5Block",
        "OPTDecoderLayer",
    ]
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        (|||)Initialize the weights()
        factor = self.config.initializer_range

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=factor)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Blip2VisionEmbeddings):
            nn.init.trunc_normal_(module.position_embedding, mean=0.0, std=factor)
            nn.init.trunc_normal_(module.class_embedding, mean=0.0, std=factor)
        elif isinstance(
            module,
            (
                Blip2Model,
                Blip2TextModelWithProjection,
                Blip2VisionModelWithProjection,
                Blip2ForConditionalGeneration,
                Blip2ForImageTextRetrieval,
            ),
        ):
            module.query_tokens.data.zero_()
        """

