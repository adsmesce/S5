X = [[2,2,3],
    [4 ,8,6],
    [7 ,8,9]]
Y = [[9,8,5],
    [6,9,4],
    [3,4,3]]
result = [[0,0,0],
    [0,0,0],
    [0,0,0]]
print("\n")
for r in X:
    print(r)
print("\n")
for r in Y:
    print(r)
for i in range(len(X)):  
    for j in range(len(X[0])):
        result[i][j] = X[i][j] + Y[i][j]
print("\n")
for r in result:
    print(r)
