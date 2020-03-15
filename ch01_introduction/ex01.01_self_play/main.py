#!/usr/bin/env python3

import numpy



class ConsolePlayer:
    def __init__(self, name):
        self.name = name


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
        while not self.finished:
            try:
                self.move(self.player_num)
            except Exception as e:
                print("Player {} caused an exception:\n{}".format(self.player_num, e))
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



def main():
    player1 = ConsolePlayer("P 1")
    player2 = ConsolePlayer("P 2")
    game = TicTacToe(player1, player2)
    game.run()



if __name__ == '__main__':
    main()
