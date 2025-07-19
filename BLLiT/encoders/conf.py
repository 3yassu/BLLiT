#conf.py
from transformers import PretrainedConfig

#PretrainedConfig allows for easy saving of configs :)

class BllitPointConfig(PretrainedConfig):
	model_type = "bllit_point_model"
	base_config_key = "point_config"
	def __init__(
		self,
		hidden_size=1408,
		intermediate_size=6144,
		num_hidden_layers=39,
		num_attention_heads=16,
		image_size=224,
		patch_size=14,
		hidden_act="gelu",
		layer_norm_eps=1e-6,
		attention_dropout=0.0,
		initializer_range=1e-10,
		qkv_bias=True,
		**kwargs,
	):
		super().__init__(**kwargs)

		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.patch_size = patch_size
		self.image_size = image_size
		self.initializer_range = initializer_range
		self.attention_dropout = attention_dropout
		self.layer_norm_eps = layer_norm_eps
		self.hidden_act = hidden_act
		self.qkv_bias = qkv_bias

class BllitVisionConfig(PretrainedConfig):
	model_type = "bllit_vision_model"
	base_config_key = "vision_config"
	def __init__(
		self,
		hidden_size=1408,
		intermediate_size=6144,
		num_hidden_layers=39,
		num_attention_heads=16,
		image_size=224,
		patch_size=14,
		hidden_act="gelu",
		layer_norm_eps=1e-6,
		attention_dropout=0.0,
		initializer_range=1e-10,
		qkv_bias=True,
		**kwargs,
	):
		super().__init__(**kwargs)

		self.hidden_size = hidden_size
		self.intermediate_size = intermediate_size
		self.num_hidden_layers = num_hidden_layers
		self.num_attention_heads = num_attention_heads
		self.patch_size = patch_size
		self.image_size = image_size
		self.initializer_range = initializer_range
		self.attention_dropout = attention_dropout
		self.layer_norm_eps = layer_norm_eps
		self.hidden_act = hidden_act
		self.qkv_bias = qkv_bias
