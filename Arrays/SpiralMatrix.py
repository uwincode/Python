# right--bottom--left--top
#         left            right
#           0  1  2  3  4  5
# 0 top     1  2  3  4  5  6
# 1         20 21 22 23 24 7
# 2         19 32 33 34 25 8
# 3         18 31 36 35 26 9
# 4         17 30 29 28 27 10
# 5bottom   16 15 14 13 1  11


def Spiral(mat):
    top = 0
    left = 0
    bottom = len(mat) -1
    right = len(mat[0])-1
    res = []
    while top<= bottom and left<=right:
        for i in range(left, right+1, 1):
            res.append(mat[top][i])
        top+=1
        for i in range(top, bottom+1, 1):
            res.append(mat[i][right])
        right-=1
        for i in range(right, left-1, -1):
            res.append(mat[bottom][i])
        bottom-=1
        for i in range(bottom, top-1, -1):
            res.append(mat[i][left])
        left+=1

    return res

matrix = [
    [1,  2,  3,  4,  5,  6],
    [20,21, 22, 23, 24, 7],
    [19,32, 33, 34, 25, 8],
    [18,31, 36, 35, 26, 9],
    [17,30, 29, 28, 27,10],
    [16,15, 14, 13, 12,11]
]

spiral = Spiral(matrix)
for val in spiral:
    print(val, end=' ')
# print()