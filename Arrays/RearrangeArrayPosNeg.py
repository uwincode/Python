# Variety - 1 --> pos==neg

def PosNeg(arr):
    n=len(arr)
    pos = 0
    neg = 1
    res=[0]*n
    for i in range(n):
        if arr[i] < 0:
            res[neg] = arr[i]
            neg +=2
        else:
            res[pos] = arr[i]
            pos +=2
    return res


# Variety 2 ---> if pos != neg then either pos > neg or pos < neg
def PosNeg(arr):
    n = len(arr)
    pos = []
    neg = []

    for i in range(n):
        if arr[i] > 0:
            pos.append(arr[i])
        else:
            neg.append(arr[i])
    while pos > neg:
        for i in range(neg):
            arr[2*i] = pos[i]
            arr[2*i+1] = neg[i]


arr = [1, -2, 3, -4, -1, 4]
print(PosNeg(arr))      
