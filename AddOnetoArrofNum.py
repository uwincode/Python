def addOne(arr):

    n = len(arr)
    # Traverse from last element
    # start from last n-1, stop at -1 (first), step by -1
    for i in range(n-1, -1, -1):
        if arr[i] < 9:
            # val i can be < 9 or = 9 and cant be > 9
            arr[i] += 1
            return arr
        
        # if val in ith position is = 9 then make it 0 and change its next val

    return [1] + [0]*n

if __name__ == "__main__":
    # res = [1, 2, 4]
    res = [9, 9, 9]
    print(addOne(res))