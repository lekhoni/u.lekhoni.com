import random

class RandomComputerPlayer:
    def __init__(self, letter):
        self.letter = letter

    def get_move(self, game):
        square = random.choice(game.available_moves())
        return square

if __name__ == '__main__':
    x_player = HumanPlayer('X')
    o_player = RandomComputerPlayer('O')
    t = TicTacToe()
    play(t, x_player, o_player, print_game=True)
