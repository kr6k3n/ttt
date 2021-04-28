from .simulator import TTTSimulator

from typing import Optional, Tuple, Union
import numpy as np
import copy
import random

callcount = int()


def horiz(board_state: np.ndarray, player: int) -> bool:
	for row in range(3):
		if board_state[row].sum() == 3*player:
			return True
	return False


def vert(board_state: np.ndarray, player: int) -> bool:
	for col in range(3):
		if board_state[:, col].sum() == 3*player:
			return True
	return False


def diagonals(board_state: np.ndarray, player: int) -> bool:
	first_diag = sum(board_state[i][i] for i in range(3)) == 3*player
	second_diag = sum(board_state[i][2-i] for i in range(3)) == 3*player
	return first_diag or second_diag


def get_winner(board_state: np.ndarray) -> Optional[int]:
	for player in (1, -1):
		if any((horiz(board_state, player),
					vert(board_state, player),
					diagonals(board_state, player))):
			return player
	if (board_state.flatten()**2).sum() == 9:  # draw
		return 0


def new_state(board_state: np.ndarray, action: Tuple[int, int], my_turn: bool) -> np.ndarray:
	new_board = copy.deepcopy(board_state)
	new_board[action[0]][action[1]] = 1 if my_turn else -1
	return new_board
# ME == 1


def memoize(f):
	memo = {}

	def helper(*args, **kwargs):
		if "get_cache" in kwargs:
			return memo
		x = "".join(map(str, args))
		if x not in memo:
			memo[x] = f(*args)
		return memo[x]
	return helper


@memoize
def minimax(board_state: np.ndarray, my_turn: bool) -> int:
	winner = get_winner(board_state=board_state)
	if not winner is None:
		return winner

	best_score = (float('-inf') if my_turn else float('inf'), (None, None))
	choice_func = max if my_turn else min

	moves = list()
	for i in range(3):
		for j in range(3):
			if board_state[i][j] == 0:
				move = minimax(new_state(board_state, action=(i, j),
												my_turn=my_turn), not my_turn)
				#print("#1",move)					
				if type(move) == int:
					move = (move, (i, j))
				else:
					while type(move) != int:
						move = move[0]
					move = (move, (i, j))
				#print("#2",move)					
				best_score = choice_func(best_score, move)
				moves.append(move)
	#print(moves, best_score)
	return [move for move in moves if move[0] == best_score[0]]


@memoize
def get_optimal_move(board_state: np.ndarray) -> Tuple[int, int]:
	moves = list()
	for i in range(3):
		for j in range(3):
			if board_state[i][j] == 0:
				moves.append(minimax(new_state(board_state, action=(i, j), my_turn=True), False))
	best_score = max(moves)
	return random.choice([move for move in moves if move[0] == best_score[0]])


def get_player_move(sim: TTTSimulator) -> Tuple[int, int]:
	move = tuple(map(int, input("enter move (i j): ").split(" ")))
	while sim.illegal_move(move[1], move[0]):
		print("invalid move")
		move = tuple(map(int, input("enter move (i j): ").split(" ")))
	return move



def initialize_optimal_move_cache() -> None:
	get_optimal_move(np.zeros(shape=(3, 3), dtype=np.int8))
	for i in range(3):
		for j in range(3):
			board_state = np.zeros(shape=(3, 3), dtype=np.int8)
			board_state[i][j] = -1
			get_optimal_move(board_state)
	print("computed all possible moves for tic tac toe")

def get_state_from_cache_index(cache_index : str) -> np.ndarray:
	cache_index = cache_index.replace("[", " ")
	cache_index = cache_index.replace("]", " ")
	cache_index = cache_index.replace("\n", " ")
	cache_index = cache_index.split(" ")
	cache_index = [int(i) for i in cache_index if i in ["-1", "1", "0"]]
	return np.array(cache_index, dtype=np.int8).reshape((3, 3))
	

initialize_optimal_move_cache()
MINIMAX_CACHE = minimax(get_cache=True)

if __name__ == '__main__':
	test: str = next(iter(MINIMAX_CACHE))
	print(MINIMAX_CACHE[test])
	print(get_state_from_cache_index(test))
