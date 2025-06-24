# Row 1: 1
# Row 2: 1 1
# Row 3: 1 (1+1) 1 = 1 2 1
# Row 4: 1 (1+2) (2+1) 1 = 1 3 3 1
# Row 5: 1 (1+3) (3+3) (3+1) 1 = 1 4 6 4 1
# Type 1: Given Row and Column, Find Element
# Problem Statement: Given r (row number) and c (column number), return the element at that position.
# Solution: Directly use the R-1 C C-1 formula, leveraging the optimized nCr function.

def ncr(n, r):
    res = 1
    for i in range(0, r, 1):
        res = res * (n-i)
        res = res // (i+1)
    return res

# Type 2: Given N, Print the Nth Row
# def Pascal(n):
#     res = 1
#     print(res)
#     for i in range(1, n, 1):
#         res = res * (n-i)
#         res = res // (i)
#         print(res)

# Type 3: Given n (number of rows), return the entire triangle as a list of lists.
def Pascal(n):
    res = []
    for row in range(1, n+1):
        temp = []
        for col in range(1, row+1):
            temp.append(ncr(row-1, col-1))
        res.append(temp)
    return res

print(Pascal(5))
    
