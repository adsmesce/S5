X = []
Y = []

rowsX = int(input("\n Enter Number of rows of First matrix : "))
colsX = int(input("\n Enter Number of cols of First matrix : "))
rowsY = int(input("\n Enter Number of rows of Second matrix : "))
colsY = int(input("\n Enter Number of cols of Second matrix : "))

if colsX != colsY:
	print("Error: The matrices cannot be Add because of the dimensions are not same!!.")
	exit()
elif rowsX != rowsY:
	print("Error: The matrices cannot be Add because of the dimensions are not same!!.")
	exit()
    
result = [[0 for i in range(colsX)] for j in range(rowsX)]

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
    for j in range(colsY):
        result[i][j] = X[i][j] + Y[i][j]

print("\nResult:")
for r in result:
    print(r)

