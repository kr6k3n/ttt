import torch
from src.model import Fully_Connected
from src.device import device
from src.simulator import TTTSimulator

from typing import Tuple

#load config
import yaml
config = None
with open(r'./config.yaml') as f:
	config = yaml.load(f, Loader=yaml.FullLoader)

input_size = 9 * 3  # can be player 1, player 2 or empty

net_description = [
    input_size,
    input_size*8,
    input_size*4,
    input_size*2,
    input_size,
    9
]

#load model to gpu
model = Fully_Connected(net_description).double().to(device)
model.load_state_dict(torch.load(config["save_path"]))
model.eval() # sets to eval mode


def get_player_move(sim: TTTSimulator) -> Tuple[int, int]:
	move = tuple(map(int, input("enter move (i j): ").split(" ")))
	while sim.illegal_move(move[1], move[0]):
		print("invalid move")
		move = tuple(map(int, input("enter move (i j): ").split(" ")))
	return move


with torch.no_grad():
  while True:
    sim = TTTSimulator()
    turn_count = int()
    game_ended = False
    while not game_ended:
      move = None
      game_state = sim.serialized_state(sim.turn)
      game_state = torch.from_numpy(game_state).double().to(device)
      if turn_count % 2 == 0:  # minimax move
        move = model(game_state).argmax()
        move = (move//3, move % 3)
      else:  # player move
        move = get_player_move(sim)
      game_ended = sim.act(col=move[1], row=move[0])
      turn_count += 1
      print(sim)

