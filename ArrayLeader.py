def arrayLeader(arr):
    res = []
    max_right = arr[-1]
    res.append(max_right)
    n = len(arr)
    for i in range(n-2, -1, -1):
        
        if arr[i] >= max_right:

            max_right = arr[i]
            res.append(max_right)

    res.reverse()
    return res

if __name__ == "__main__":
    arr = [16, 17, 4, 3, 5, 2]
    res = arrayLeader(arr)
    print(" ".join(map(str, res)))
