class NQueens:
  def __init__(self, size):
    self.n = size
    self.__reset_board()

  def __reset_board(self):
    self.board = [[0 for _ in range(self.n)] for _ in range(self.n)]

  def display(self):
    for i in range(self.n):
      for j in range(self.n):
        print('Q', end=' ') if self.board[i][j] else print('_', end=' ')
      print()

  def __is_valid(self, queens, idx):
    for i in range(idx):
      if queens[i] == queens[idx]: return False
      elif i + queens[i] == idx + queens[idx]: return False
      elif i - queens[i] == idx - queens[idx]: return False
    return True

  def solve_backtracking(self):
    self.__reset_board()
    queens = [0 for _ in range(self.n)]
    idx = 1
    while idx < self.n:
      found = False
      for i in range(queens[idx], self.n):
        queens[idx] = i
        if self.__is_valid(queens, idx):
          found = True
          break
      if not found:
        while queens[idx] == self.n-1:
          queens[idx] = 0
          idx -= 1
          if idx == -1: break
        if idx == -1: break
        else: queens[idx] += 1
      else: idx += 1
    if idx == -1: print('NO SOLUTION.')
    else:
      for i in range(self.n):
        self.board[i][queens[i]] = 1
      print('SOLUTION FOUND.')
      self.display()

  def __is_valid_bnb(self, queens, idx, cols, d_back, d_for):
    if cols[queens[idx]]: return False
    if d_back[idx-queens[idx]+self.n-1]: return False
    if d_for[idx+queens[idx]]: return False
    return True

  def __set_arrays(self, idx, queens, cols, d_back, d_for, val):
    cols[queens[idx]] = val
    d_back[idx-queens[idx]+self.n-1] = val
    d_for[idx+queens[idx]] = val

  def solve_bnb(self):
    self.__reset_board()
    queens = [0 for _ in range(self.n)]
    cols = [False for _ in range(self.n)]
    d_back = [False for _ in range(2*self.n-1)]
    d_for = [False for _ in range(2*self.n-1)]
    idx = 1
    cols[0] = d_back[self.n-1] = d_for[0] = True
    while idx < self.n:
      found = False
      for i in range(queens[idx], self.n):
        queens[idx] = i
        if self.__is_valid_bnb(queens, idx, cols, d_back, d_for):
          found = True
          self.__set_arrays(idx, queens, cols, d_back, d_for, True)
          break
      if not found:
        queens[idx] = 0
        idx -= 1
        if idx == -1: break
        else:
          self.__set_arrays(idx, queens, cols, d_back, d_for, False)
          queens[idx] += 1
      else: idx += 1
    if idx == -1: print('NO SOLUTION.')
    else:
      for i in range(self.n):
        self.board[i][queens[i]] = 1
      print('SOLUTION FOUND.')
      self.display()

import time

n_queens = NQueens(20)

start = time.perf_counter()
n_queens.solve_backtracking()
end = time.perf_counter()
print(end-start)

"""
SOLUTION FOUND.
Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ 
_ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ 
_ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ 
_ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ 
10.804984268000226
"""

start = time.perf_counter()
n_queens.solve_bnb()
end = time.perf_counter()
print(end-start)

"""
SOLUTION FOUND.
Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ 
_ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ Q _ 
_ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ 
_ _ _ _ _ Q _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ _ _ Q _ _ _ _ _ _ _ _ _ 
1.795702335999522
"""