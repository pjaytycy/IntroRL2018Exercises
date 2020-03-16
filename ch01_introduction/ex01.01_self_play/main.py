#!/usr/bin/env python3

import numpy
import random
import traceback



class ConsolePlayer:
    def __init__(self, name):
        self.name = name


    def start(self, player_num):
        self.name = input("Enter the name for player {} :".format(player_num))


    def get_move(self, grid):
        print("Move for {}".format(self.name))
        print(str(grid))
        row = int(input("enter row (0, 1 or 2) :"))
        col = int(input("enter column (0, 1 or 2) :"))
        return row, col


    def won(self):
        print("{} WON!".format(self.name))


    def lost(self):
        print("{} LOST!".format(self.name))


    def draw(self):
        print("{} -> draw".format(self.name))



class RLPlayer:
    def __init__(self, name, explore_pct):
        self.name = name
        self.explore_pct = explore_pct
        self.alpha = 0.5
        self.game_num = 0
        self.num_grid_states = 3**9-1
        # 9 squares that have either value 0, 1 or 2. No need to split between players, as 50% of the states
        # will only be valid for player1 and 50% will only be valid for player2. If you get a random grid, it
        # is clear whose turn it is.
        self.values = numpy.ones(self.num_grid_states, float) * 0.5
        self.debug = True
        self.debug_extra = False


    def start(self, player_num):
        self.prev_game_state = None
        self.next_game_state = None
        self.player_num = player_num


    def get_move(self, grid):
        if self.debug:
            print("Move for {}".format(self.name))
            print(str(grid))
        row, col = self.select_move(grid)
        if self.debug:
            print("Selected {}, {}".format(row, col))
        return row, col


    def grid_to_state(self, grid):
        return numpy.sum(grid.flat * 3**numpy.arange(8, -1, -1))


    def select_move(self, grid):
        self.prev_game_state = self.next_game_state
        next_moves, next_game_states = self.get_all_possible_moves(grid)
        if len(next_moves) == 0:
            raise Exception("No more moves possible")

        rand_val = random.random()
        self.greedy_move = False
        if rand_val > self.explore_pct:
            self.greedy_move = True
            next_moves, next_game_states = self.select_greedy_moves(next_moves, next_game_states)

        move = self.select_random_move(next_moves, next_game_states)
        if self.greedy_move:
            self.update_value()

        return move


    def update_value(self):
        if self.debug_extra:
            print("update_value : prev state = {}, next state = {}".format(self.prev_game_state, self.next_game_state))

        if self.prev_game_state is None or self.next_game_state is None:
            return

        prev_value = self.values[self.prev_game_state]
        next_value = self.values[self.next_game_state]
        diff_value = next_value - prev_value
        new_prev_value = prev_value + self.alpha * diff_value
        if self.debug_extra:
            print("prev value = {}, next value = {}, diff value = {}, new_prev_value = {}".format(
                prev_value, next_value, diff_value, new_prev_value))
        self.values[self.prev_game_state] = new_prev_value


    def select_greedy_moves(self, next_moves, next_game_states):
        best_moves = [next_moves[0]]
        best_game_states = [next_game_states[0]]
        best_value = self.values[best_game_states[0]]
        if self.debug_extra:
            print("select_greedy_move out of {} options".format(len(next_moves)))
            print("{} : {} -> {}".format(best_value, best_moves[0], best_game_states[0]))
        for i in range(1, len(next_moves)):
            value = self.values[next_game_states[i]]
            if self.debug_extra:
                print("{} : {} -> {}".format(value, next_moves[i], next_game_states[i]))
            if value == best_value:
                best_moves.append(next_moves[i])
                best_game_states.append(next_game_states[i])
            elif value > best_value:
                best_moves = [next_moves[i]]
                best_game_states = [next_game_states[i]]
                best_value = value
        return best_moves, best_game_states


    def select_random_move(self, next_moves, next_game_states):
        i = random.randint(0, len(next_moves) - 1)
        self.next_game_state = next_game_states[i]
        return next_moves[i]


    def get_all_possible_moves(self, grid):
        next_moves = []
        next_game_states = []
        for row in range(3):
            for col in range(3):
                if grid[row, col] == 0:
                    next_moves.append((row, col))
                    tmp_grid = numpy.copy(grid)
                    tmp_grid[row, col] = self.player_num
                    next_game_states.append(self.grid_to_state(tmp_grid))
        return next_moves, next_game_states


    def won(self):
        print("{} WON!".format(self.name))
        self.game_finished(+1)


    def lost(self):
        print("{} LOST!".format(self.name))
        self.game_finished(-1)


    def draw(self):
        print("{} -> draw".format(self.name))
        self.game_finished(0)


    def game_finished(self, reward):
        self.game_num += 1
        orig_value = self.values[self.next_game_state]
        new_value = reward # orig_value + reward
        self.values[self.next_game_state] = new_value
        if self.debug:
            print("Changing value for {} from {} to {}".format(self.next_game_state, orig_value, new_value))



class TicTacToe:
    def __init__(self, player1, player2):
        self.grid = numpy.zeros((3, 3), int)
        self.player1 = player1
        self.player2 = player2
        self.winner = None
        self.finished = False
        self.player_num = 1


    def switch_player_num(self):
        if self.player_num == 1:
            self.player_num = 2
        elif self.player_num == 2:
            self.player_num = 1
        else:
            raise Exception("Invalid player_num : {}".format(self.player_num))


    def run(self):
        self.player1.start(1)
        self.player2.start(2)
        while not self.finished:
            try:
                self.move(self.player_num)
            except Exception as e:
                print("Player {} caused an exception:\n{}".format(self.player_num, e))
                traceback.print_exc()
                self.finished = True
                self.switch_player_num()
                self.winner = self.player_num
                break
            self.switch_player_num()
            self.check_state()

        if self.winner == 1:
            self.player1.won()
            self.player2.lost()
        elif self.winner == 2:
            self.player2.won()
            self.player1.lost()
        else:
            self.player1.draw()
            self.player2.draw()


    def move(self, player_num):
        player = None
        if player_num == 1:
            player = self.player1
        elif player_num == 2:
            player = self.player2
        else:
            raise Exception("Invalid player number : {}".format(player_num))

        row, col = player.get_move(numpy.copy(self.grid))
        ok = (self.grid[row, col] == 0)
        if not ok:
            raise Exception("Grid position not valid : row {}, col {} in grid:\n{}".format(row, col, str(self.grid)))

        self.grid[row, col] = player_num


    def check_state(self):
        row = 0
        while row < 3 and not self.finished:
            self.check_row(row)
            row += 1

        col = 0
        while col < 3 and not self.finished:
            self.check_col(col)
            col += 1

        if not self.finished:
            self.check_diag(0, +1)

        if not self.finished:
            self.check_diag(2, -1)

        if not self.finished:
            self.check_empty()


    def check_row(self, row):
        p0, p1, p2 = self.grid[row, :]
        self.check_3v(p0, p1, p2)


    def check_col(self, col):
        p0, p1, p2 = self.grid[:, col]
        self.check_3v(p0, p1, p2)


    def check_diag(self, col0, col_delta):
        p0 = self.grid[0, col0 + 0 * col_delta]
        p1 = self.grid[1, col0 + 1 * col_delta]
        p2 = self.grid[2, col0 + 2 * col_delta]
        self.check_3v(p0, p1, p2)


    def check_3v(self, p0, p1, p2):
        if p0 == p1 == p2 and p0 != 0:
            self.winner = p0
            self.finished = True


    def check_empty(self):
        cells_played = numpy.count_nonzero(self.grid)
        if cells_played == 9:
            self.finished = True



def main_console():
    player1 = ConsolePlayer("player1")
    player2 = ConsolePlayer("player2")
    game = TicTacToe(player1, player2)
    game.run()



def main_rl():
    player1 = RLPlayer("player_greedy", explore_pct = 0.10)
    player2 = RLPlayer("player_exporer", explore_pct = 0.25)
    player1.debug = False
    player2.debug = False
    for g in range(10000):
        print("game {}".format(g))
        game = TicTacToe(player1, player2)
        game.run()
        game = TicTacToe(player2, player1)
        game.run()
    player1.debug = True
    player2.debug = True
    player1.debug_extra = True
    player2.debug_extra = True
    game = TicTacToe(player1, player2)
    game.run()
    game = TicTacToe(player2, player1)
    game.run()
    player_me = ConsolePlayer("PJ")
    player1.explore_pct = 0.0
    player2.explore_pct = 0.0
    game = TicTacToe(player1, player_me)
    game.run()
    game = TicTacToe(player_me, player1)
    game.run()



if __name__ == '__main__':
    main_rl()
