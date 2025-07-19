#BLLIT_proc.py
#Processor class for BLLiT
#
#
#BLLiT_proc.py
from typing import Optional, Union
from transformers import BatchFeature
#from ..image_point_utils import ImageInput, PointInput (commented for bug testing)
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack #For Now I believe I can get away with using these, Will remove later
from transformers.tokenization_utils_base import AddedToken, BatchEncoding, PreTokenizedInput, TextInput

#---
import numpy as np
PointInput = Union[
	np.ndarray, "torch.Tensor", list[np.ndarray], list["torch.Tensor"] #"PointInput", list["PointInput"],
]  # noqa
ImageInput = Union[
	"PIL.Image.Image", np.ndarray, "torch.Tensor", list["PIL.Image.Image"], list[np.ndarray], list["torch.Tensor"]
]  # noqa
#---
class BllitProcessorKwargs(ProcessingKwargs, total=False):
	_defaults = {
		"text_kwargs": {
			"add_special_tokens": True,
			"padding": False,
			"stride": 0,
			"return_overflowing_tokens": False,
			"return_special_tokens_mask": False,
			"return_offsets_mapping": False,
			"return_token_type_ids": False,
			"return_length": False,
			"verbose": True,
		},
		"images_kwargs": {},
		"points_kwargs": {},
	}

class BllitProcessor(ProcessorMixin):
	"""
	The purpose of the BllitProcessor class is to wrap an image and point cloud processor and a tokenizer into a single processor
	Args:
		image_processor (`BllitImageProcessor`, *optional*): - - - - - - - - - - - - - - -|
			An instance of [`BllitImageProcessor`]. An optional input.                    | One of these are required.
		point_processor (`BllitPointProcessor`, *optional*):                              | Though both may be used.
			An instance of [`BllitPointProcessor`]. An optional input. - - - - - - - - - -|
		tokenizer (`AutoTokenizer`):
			An instance of [`PreTrainedTokenizer`].  The tokenizer is a required input.
		num_query_tokens (`int`, *optional*):
			Number of tokens used by the Qformer as queries, should be same as in model's config.
	"""
	attributes = ["image_processor", "point_processor", "tokenizer"]
	image_processor_class = ("BlipImageProcessor", "BlipImageProcessorFast") #make own later
	point_processor_class = ("BllitPointProcessor") #may remove later
	tokenizer_class = "AutoTokenizer"
	def __init__(self, tokenizer, image_processor=None, point_processor=None, num_query_tokens=None, **kwargs): #I put tokenizer before (image&point)_processor, I might have to switch back if issues happen
		tokenizer.return_token_types_ids = False
		if image_processor is None and point_processor is None:
			raise ValueError("Invalid parameters, include either an image processor or a point processor.")
		if image_processor:
			self.current_image_processor = image_processor
			if not hasattr(tokenizer, "image_token"):
				self.image_token = AddedToken("<image>", normalized=False, special=True)
				tokenizer.add_tokens([self.image_token], special_tokens=True)
			else:
				self.image_token = tokenizer.image_token
		if point_processor:
			self.current_point_processor = point_processor
			if not hasattr(tokenizer, "point_token"):
				self.point_token = AddedToken("<point>", normalized=False, special=True)
				tokenizer.add_tokens([self.point_token], special_tokens=True)
			else:
				self.point_token = tokenizer.point_token
		self.num_query_tokens = num_query_tokens
		if image_processor is not None and point_processor is not None:
			super().__init__(image_processor, point_processor, tokenizer)
		elif point_processor is not None:
			super().__init__(point_processor, tokenizer)
		else:
			super().__init__(image_processor, tokenizer)
	def __call__(self,
		images: ImageInput = None,
		points: PointInput = None,
		text: Optional[Union[str, list[str], TextInput, PreTokenizedInput]] = None,
		audio=None,
		videos=None,
		**kwargs: Unpack[BllitProcessorKwargs],
	) -> BatchEncoding:
		"""
		The purpose of this class is to apply the inputs to the respective processors, these inputs will then be able to go to some BllitModel class
		Args:
			images (`ImageInput`):
				image(s) to be prepared.
			points (`PointInput`):
				point cloud(s) to be prepaered.
			text (`TextInput`, `PreTokenizedInput`, `list[TextInput]`, `list[PreTokenizedInput]`):
				The sequence or batch of sequences to be encoded.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                    - `'tf'`: Return TensorFlow `tf.constant` objects.
                    - `'pt'`: Return PyTorch `torch.Tensor` objects.
                    - `'np'`: Return NumPy `np.ndarray` objects.
                    - `'jax'`: Return JAX `jnp.ndarray` objects.
		"""
		if images is None and points is None and test is None:
			raise ValueError("You have to specify either images, point clouds, or text.")
		output_kwards = self._merge_kwards(
			BllitProcessorKwargs,
			tokenizer_init_kwargs=self.tokenizer.init_kwargs,
			**kwargs,
		)
		# BC for explicit return_tensors
		if "return_tensors" in output_kwargs["common_kwargs"]:
			return_tensors = output_kwargs["common_kwargs"].pop("return_tensors", None)
		else:
			return_tensors = None
		encoding = BatchFeature(tensor_type=return_tensors)
		if text is not None:
			if isinstance(text, str):
				text = [text]
			elif not isinstance(text, list) and not isinstance(text[0], str):
				raise ValueError("Invalid text input, provide a list of strings or a string")
			text_encoding = {}
			return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
			_text_encoding = self.tokenizer(text, **output_kwargs["text_kwargs"], return_tensors=None)
			output_kwargs["text_kwargs"]["return_tensors"] = return_tensors

			# if we know how many query tokens, expand text inside processor. We need this hacky manipulation - | comments from copy
			# because BLIP expects image tokens to be at the beginning even before BOS token - - - - - - - - - -| :P
			if self.num_query_tokens is not None:
				image_tokens = self.image_token.content * self.num_query_tokens
				point_tokens = self.point_tokens.content * self.num_query_tokens
				image_token_encoding = self.tokenizer(
					[image_tokens] * len(text), add_special_tokens=False, return_tensors=None
				)
				point_token_encoding = self.tokenizer(
					[point_tokens] * len(text), add_special_tokens=False, return_tensors=None
				)
				for k in _text_encoding:
					text_encoding[k] = [
						img_encoding + txt_encoding
						for img_encoding, txt_encoding in zip(image_token_encoding[k], _text_encoding[k])
					]
			else:
				text_encoding = _text_encoding
				print(
					"Expanding inputs for image tokens in BLIP-2 should be done in processing. ",
					"Please follow instruction here (https://gist.github.com/zucchini-nlp/e9f20b054fa322f84ac9311d9ab67042) to update your BLIP-2 model. ",
					"Using processors without these attributes in the config is deprecated and will throw an error in v4.50."
				)

			encoding.update(BatchEncoding(text_encoding, tensor_type=return_tensors))

		if images is not None:
			image_encoding = self.image_processor(images, **output_kwargs["images_kwargs"])
			encoding.update(image_encoding)
		if points is not None:
			point_encoding = self.point_processor(point, **output_kwargs["points_kwargs"])
			encoding.update(point_encoding)
		return encoding
	# Copied from transformers.models.blip.processing_blip.BlipProcessor.batch_decode with BertTokenizerFast->PreTrainedTokenizer
	def batch_decode(self, *args, **kwargs):
		"""
		This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
		refer to the docstring of this method for more information.
		"""
		return self.tokenizer.batch_decode(*args, **kwargs)

	# Copied from transformers.models.blip.processing_blip.BlipProcessor.decode with BertTokenizerFast->PreTrainedTokenizer
	def decode(self, *args, **kwargs):
		"""
		This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer to
		the docstring of this method for more information.
		"""
		return self.tokenizer.decode(*args, **kwargs)

	def model_input_names(self):
		tokenizer_input_names = self.tokenizer.model_input_names
		image_processor_input_names = self.image_processor.model_input_names
		point_processor_input_names - self.point_processor.model_input_names
		return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names + point_processor_input_names))

__all__ = ["BllitProcessor"]
