def subset(idx, total, res, arr):
    n = len(arr)
    if idx == n:
        res.append(total)
        return
    
    subset(idx + 1, total + arr[idx], res, arr)
    subset(idx + 1, total, res, arr)
    

if __name__ == "__main__":
    arr = [3, 1, 2]
    res = []
    subset(0, 0, res, arr)
    res.sort()
    print("The sum of each subset is")
    for s in res:
        print(s, end=" ")
    print()