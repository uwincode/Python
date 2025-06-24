# First pass: Find all zeros in the matrix and mark their entire row and column with -1 (except existing zeros)
# Second pass: Convert all -1 markers back to 0
# def SetMatrix(mat):
#     n = len(mat)
#     m = len(mat[0])
#     for i in range(n):
#         for j in range(m):
#             if mat[i][j]==0:
#                 markrow(mat, i)
#                 markcol(mat, j)
#     for i in range(n):
#         for j in range(m):
#             if mat[i][j]==-1:
#                 mat[i][j] = 0
#     return mat
            
# def markrow(mat, i):
#     for j in range(len(mat[0])):
#         if mat[i][j] != 0:
#             mat[i][j] = -1
#     return mat

# def markcol(mat, j):
#     for i in range(len(mat)):
#         if mat[i][j] != 0:
#             mat[i][j] = -1
#     return mat

# better soln
def SetMatrix(mat):
    n = len(mat)
    m = len(mat[0])
    col = [0] * m  # Create array of size m, initialized with zeros
    row = [0] * n
    for i in range(n):
        for j in range(m):
            if mat[i][j]==0:
                row[i] = 1
                col[j] = 1
    for i in range(n):
        for j in range(m):
            if row[i] or col[j]:
                mat[i][j] = 0
    return mat



matrix = [
    [1, 2, 3],
    [4, 0, 6],
    [7, 8, 9]
]

result = SetMatrix(matrix)
for row in result:
    print(row)

