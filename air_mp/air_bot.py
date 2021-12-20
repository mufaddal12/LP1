class Game:
  def __init__(self):
    self.board = [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]]
    self.turn = 0
  
  def display(self):
    for i in range(3):
      for j in range(3):
        if self.board[i][j] == -1: print('_', end=' ')
        else:
          print('X',end=' ') if self.board[i][j] == 1 else print('O',end=' ')
      print()

  def is_valid(self, i, j):
    return self.board[i][j] == -1

  def no_moves(self):
    for i in range(3):
      for j in range(3):
        if self.board[i][j] == -1: return False
    return True

  def move(self, player, i, j):
    if self.board[i][j] == -1:
      self.board[i][j] = player
      return True
    return False

  def remove(self, player, i, j):
    if self.board[i][j] == player:
      self.board[i][j] = -1
      return True
    return False

  def winner(self):
    for i in range(3):
      if self.board[i][0] == self.board[i][1] and self.board[i][0] == self.board[i][2]:
        return self.board[i][0]
      elif self.board[0][i] == self.board[1][i] and self.board[2][i] == self.board[0][i]:
        return self.board[0][i]
    if self.board[0][0] == self.board[1][1] and self.board[1][1] == self.board[2][2]:
      return self.board[0][0]
    elif self.board[2][0] == self.board[1][1] and self.board[1][1] == self.board[0][2]:
      return self.board[1][1]
    return -1

class Bot:
  def __init__(self, player):
    self.player = player
    self.board = Game()

  def value(self):
    if self.board.winner() == self.player: return 1
    elif self.board.winner() == 1 - self.player: return -1
    else: return 0

  def make_move(self):
    best_move = None
    max_val = -100
    for i in range(3):
      for j in range(3):
        if self.board.is_valid(i,j):
          self.board.move(self.player,i,j)
          if self.value() == 1 or self.board.no_moves():
            return (i,j)
          val = self.minimax(False)
          if val > max_val:
            best_move = (i,j)
            max_val = val
          self.board.remove(self.player,i,j)
    self.board.move(self.player, best_move[0], best_move[1])
    return best_move
  
  def minimax(self, maximizer):
    if maximizer:
      # best_move = None
      max_val = -100
      for i in range(3):
        for j in range(3):
          if self.board.is_valid(i,j):
            self.board.move(self.player,i,j)
            if self.value() == 1:
              self.board.remove(self.player,i,j)
              return 1
            elif self.board.no_moves():
              self.board.remove(self.player,i,j)
              return 0
            else:
              val = self.minimax(False)
              if val > max_val:
                max_val = val
                # best_move = (i,j)
            self.board.remove(self.player,i,j)
      return max_val

    else:
      min_val = 100
      for i in range(3):
        for j in range(3):
          if self.board.is_valid(i,j):
            self.board.move(1-self.player,i,j)
            if self.value() == -1:
              self.board.remove(1-self.player,i,j)
              return -1
            elif self.board.no_moves():
              self.board.remove(1-self.player,i,j)
              return 0
            else:
              val = self.minimax(True)
              if val < min_val:
                min_val = val
                # best_move = (i,j)
            self.board.remove(1-self.player,i,j)
      return min_val

  def opp_move(self, i, j):
    self.board.move(1 - self.player, i, j)

game = Game()
bot = Bot(1)

while not game.no_moves() and game.winner() == -1:
  print("Your Move: ", end=' ')
  my_move = input().split(',')
  i, j = int(my_move[0]), int(my_move[1])
  game.move(0, i, j)
  bot.opp_move(i, j)
  game.display()
  print()
  if game.winner() != -1 or game.no_moves():
    break

  move = bot.make_move()
  print(f"Bot's Move: {move[0]},{move[1]}")
  game.move(1, move[0], move[1])
  game.display()
  print()   