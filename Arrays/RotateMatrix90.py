# [0][0] --> [0][3]     [1][0] --> [0][2]
# [0][1] --> [1][3]     [1][1] --> [1][2]
# [0][2] --> [2][3]     [1][2] --> [2][2]
# [0][3] --> [3][3]     [1][3] --> [3][2]
#   0 to 3             1 to 2 1 to n-1 - i

# for i = 0 j = 3 then i from 1 to n-1
# for i = 1 j = 2 then i from 1 to (n-1)-i

# better soln
# def RotateMatrix(mat):
#     n = len(mat)
#     m = len(mat[0])
#     rotated = [[0] *n for _ in range(m)]
#     for i in range(n):
#         for j in range(m):
#             rotated[j][n-1-i] = mat[i][j]

#     return rotated

# Optimal soln : no extra space...finish it in the same matrix
def RotateMatrix(mat):
    n = len(mat)
    m = len(mat[0])
    # no 'rotated'..to optimise
    # transpose the matrix---> swap then reverse every row
    # wach out for i, j limits
    for i in range(0, n-1):
        for j in range(i+1, n):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    for i in range(n):
        mat[i].reverse()
    return mat


matrix = [
    [1, 2, 3],
    [4, 0, 6],
    [7, 8, 9]
]
result = RotateMatrix(matrix)
for row in result:
    print(row)