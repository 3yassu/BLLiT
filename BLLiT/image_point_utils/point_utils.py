#point_utils.py
from typing import Union
from typing import TYPE_CHECKING
import numpy as np
if TYPE_CHECKING:
	#Some point input
	import torchs
	from .point import Point

PointInput = Union[
    "BLLiT_2_redux.point.Point", np.ndarray, "torch.Tensor", list["BLLiT_2_redux.point.Point"], list[np.ndarray], list["torch.Tensor"]
]  # noqa

class BasePointProcessor:
	def __init__():
		print("Hi, my name is snapple")
