from typing import List
import torch.nn as nn
import torch.nn.functional as F


class Fully_Connected(nn.Module):
	def __init__(self, description : List[int]):
		super(Fully_Connected, self).__init__()
		layers = list()
		for i in range(len(description)-1):
			in_size, out_size = description[i], description[i+1]
			layers.append(nn.Linear(in_size, out_size))
			layers.append(nn.ReLU())
		layers.pop(-1)
		layers.append(nn.Softmax())
		self.layers = nn.ModuleList(layers)

	def forward(self, in_vec):
		out_vec = in_vec
		for layer in self.layers:
			out_vec = layer(out_vec)
		return out_vec
