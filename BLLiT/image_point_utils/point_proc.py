#point_proc.py
import os
import torch
import torch.nn as nn
import numpy as np
from typing import (Union, Optional)

class BllitPointProcessor:
	def __init__(
		self,
		root: str = "./",
		num_point: int,
		do_normalize: bool = False,
		is_normal: bool = False,
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Idk if thhis is neccessary :(
	):
		self.device = device
		"""
		Sets up everythingsies
		"""
		self.root = root

	@staticmethod
	def farthest_point_sample(point, npoint):
		"""
		Input:
			xyz: pointcloud data, [N, D]
			npoint: number of samples
		Return:
			centroids: sampled pointcloud index, [npoint, D]
		"""
		N, D = point.shape
		xyz = point[:,:3]
		centroids = np.zeros((npoint,))
		distance = np.ones((N,)) * 1e10
		farthest = np.random.randint(0, N)
		for i in range(npoint):
			centroids[i] = farthest
			centroid = xyz[farthest, :]
			dist = np.sum((xyz - centroid) ** 2, -1)
			mask = dist < distance
			distance[mask] = dist[mask]
			farthest = np.argmax(distance, -1)
		point = point[centroids.astype(np.int32)]
		return point

	def _from_file(self, name: str, file_exten: str = "txt", npoints: Optional[int] = None): #npy, txt, ply
		uniform = 0 #For now
		if file_exten == "txt":
			point_set = np.loadtxt((self.root + name), delimiter=',').astype(np.float32)
		elif file_exten == "npy":
			point_set = np.loadtxt(self.root + name).astype(np.float32)
		else:
			raise ValueError("File extention not supported, try supported file-type (txt, ply, npy)") #not ply yet lol

		if uniform and npoints is not None:
			point_set = farthest_point_sample(point_set, self.npoints)
		else:
			point_set = point_set[0:npoints, :]
		return point_set

	def _from_dir(self, name: str, file_exten: str = "txt"): #think about npoints and uniform lol
		file_list = open(self.root + name, "r").read().split("\n")
		points_list = []
		for file in file_list:
			points_list.append(torch.from_numpy(self._from_file((name+"/"+file), file_exten)).float().to(self.device))
		return points_list

	def __call__(self, name: str, file_exten: str = "txt"): #This may be the command I use to split up training and testing data.
		if os.path.isdir(self.root + "/" + name):
			return self._from_dir(name, file_exten)
		else:
			return [torch.from_numpy(self._from_file(name, file_exten)).float().to(self.device)]
__all__ = ["BllitPointProcessor"]
