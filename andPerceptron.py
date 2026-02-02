inputs=[(0,0),(0,1),(1,0),(1,1)]

def andPerceptron( x1, x2):
  w1, w2 = 1, 1
  b = -1.5

  z = w1*x1 + w2*x2 + b

  return 1 if z>=0 else 0

for x in inputs:
  print(x, "->", andPerceptron(x[0], x[1]))

