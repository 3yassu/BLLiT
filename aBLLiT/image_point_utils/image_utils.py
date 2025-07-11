#image_utils.py
from typing import Union
from typing import TYPE_CHECKING
import numpy as np 
if TYPE_CHECKING:
	import .Image
	import torch

ImageInput = Union[
    ".Image", np.ndarray, "torch.Tensor", list[".Image"], list[np.ndarray], list["torch.Tensor"]
]

class BllitImageProcessor():
	def __init__():
		print("Hi, my name is snapple")
		
class BllitImageProcessorFast():
	def __init__():
		print("Hi, my name is snapple")
