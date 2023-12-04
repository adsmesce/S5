X = []
Y = []
result = []

rowsX = int(input("\n Enter Number of rows of First matrix : "))
colsX = int(input("\n Enter Number of cols of First matrix : "))
rowsY = int(input("\n Enter Number of rows of Second matrix : "))
colsY = int(input("\n Enter Number of cols of Second matrix : "))

if rowsX != colsY:
    print("Error: The matrices cannot be multiplied because the number of columns in the first matrix is not equal to the number of rows in the second matrix.")
    exit()
    
print("Enter the values of the first matrix:")
for i in range(rowsX):
    col = []
    for j in range(colsX):
        no = int(input("Index {}{}:".format(i, j)))
        col.append(no)
    X.append(col)

print("Enter the values of the second matrix:")
for i in range(rowsY):
    col = []
    for j in range(colsY):
        no = int(input("Index {}{}:".format(i, j)))
        col.append(no)
    Y.append(col)

for i in range(rowsX):
    result.append([])
    for j in range(colsY):
        sum = 0
        for k in range(colsX):
            sum += X[i][k] * Y[k][j]
        result[i].append(sum)

print("\nResult:")
for r in result:
    print(r)

