def missRange(arr, lower, upper):
    n=len(arr)
    res =[]

    if arr[0]> lower:
        res.append([lower, arr[0]-1])
    
    for i in range(n-1):
        if arr[i+1] - arr[i] >1:
            res.append([arr[i]+1, arr[i+1]-1])

    if arr[-1]<upper:
        res.append([arr[-1]+1, upper])

    return res
if __name__ == "__main__":
    lower = 10
    upper = 50
    arr = [14, 15, 20, 30, 31, 45]
    res = missRange(arr, lower, upper)
    for v in res:
        print(v[0], v[1])