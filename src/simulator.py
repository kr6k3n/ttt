from dataclasses import dataclass, field

from typing import Union

import numpy as np


def int_to_player(i: int) -> str:
  return "o|" if i == 1 else ("x|" if i == -1 else (" |" if i == 0 else ""))


@dataclass
class TTTSimulator:
  state: np.ndarray = field(default_factory=(lambda: np.zeros(shape=(3, 3), dtype=np.int8)))
  turn = "X"
  winner = None
  game_length = 0
  game_error = False

  def __repr__(self):
    return "  |" + "\n  |".join("".join(map(int_to_player, list(self.state[row]))) for row in range(3)) + "\n"

  @property
  def next_turn(self) -> str:
    return "X" if self.turn == "O" else "O"

  def illegal_move(self, col, row) -> bool:
    return self.state[row][col] != 0

  def play(self, col: int, row: int) -> None:
    self.state[row][col] = 1 if self.turn == "O" else -1
    self.turn = "X" if self.turn == "O" else "O"

  def horiz(self, player: int) -> bool:
    for row in range(3):
      if self.state[row].sum() == 3*player:
        return True
    return False

  def vert(self, player: int) -> bool:
    for col in range(3):
      if self.state[:, col].sum() == 3*player:
        return True
    return False

  def diagonals(self, player: int) -> bool:
    first_diag = sum(self.state[i][i] for i in range(3)) == 3*player
    second_diag = sum(self.state[i][2-i] for i in range(3)) == 3*player
    return first_diag or second_diag

  def get_winner(self) -> Union[str, None]:
    for player in ["X", "O"]:
      player_val = 1 if player == "X" else -1
      if any((self.horiz(player_val), self.vert(player_val), self.diagonals(player_val))):
        return player
    return None

  def draw(self) -> bool:
    for row in range(3):
      for col in range(3):
        if self.state[row][col] == 0:
          return False
    return True

  def act(self, col, row) -> bool:
    """returns True if game ended"""
    if self.illegal_move(col, row):
      self.winner = self.next_turn
      self.game_error = True
      return True
    self.play(col, row)
    self.game_length += 1
    winner = self.get_winner()
    if not winner is None:
      self.winner = winner
      return True
    elif self.draw():
      self.winner = None
      return True
    return False

  def serialized_state(self, player_id) -> np.ndarray:
    state_copy = self.state.flatten() * (1 if player_id == "X" else -1)
    return np.concatenate(((state_copy == 1).astype(np.uint8), (state_copy == -1).astype(np.uint8), (state_copy == 0).astype(np.uint8)))
