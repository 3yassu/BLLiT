#point_utils.py
from typing import Union
from typing import TYPE_CHECKING
import numpy as np 
if TYPE_CHECKING:
	#Some point input
	import torchs

PointInput = Union[
    "PointInput", np.ndarray, "torch.Tensor", list["PointInput"], list[np.ndarray], list["torch.Tensor"]
]  # noqa

class BllitPointProcessor():
	def __init__():
		print("Hi, my name is snapple")
		
class BllitPointProcessorFast():
	def __init__():
		print("Hi, my name is snapple")
