import torch
import numpy as np
from typing import Tuple, List
from .minimax import MINIMAX_CACHE, get_state_from_cache_index
from .device import device


def convert_to_nn_output(cache : List[Tuple[int, Tuple[int]]]) -> np.ndarray:
  out = np.zeros(9, dtype=np.int8)
  for move in cache:
    i, j = move[1]
    nn_out_index = 3*i + j
    out[nn_out_index] = 1
  return out

def get_serialized_state_from_cache_index(state_index):
  state = get_state_from_cache_index(state_index).flatten()
  return np.concatenate(((state == 1).astype(np.uint8), (state == -1).astype(np.uint8), (state == 0).astype(np.uint8)))

def build_dataset() -> List[Tuple[np.ndarray, np.ndarray]]:
  dataset = list()
  for state_index in MINIMAX_CACHE.keys():
    if state_index.endswith("True") and type(MINIMAX_CACHE[state_index]) != int:
      dataset.append(
        (get_serialized_state_from_cache_index(state_index),
         convert_to_nn_output(MINIMAX_CACHE[state_index]))
      )
  return dataset


def load_dataset_to_gpu(dataset: List[Tuple[np.ndarray, np.ndarray]]):
  for i in range(len(dataset)):
    state, output = dataset[i]
    state = torch.from_numpy(state).double().to(device)
    output = torch.from_numpy(output).double().to(device)
    dataset[i] = (state, output)
  return dataset

