def MissRepeat(arr):
    # temp = []
    n = len(arr)
    freq = [0] * (n+1)
    repeat = -1
    miss = -1
    for i in range(n):
        freq[arr[i]] +=1
    for i in range(1, n+1):
        if freq[i] ==0 :
            miss = i
        if freq[i] == 2:
            repeat = i
    return [repeat, miss]


if __name__ == "__main__":
    arr = [3, 1, 3]
    ans = MissRepeat(arr)

    print(ans[0], ans[1])