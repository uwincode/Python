# [3, 1, 2]  = [], [3], [1], [2], [3, 1], [1, 2], [3, 1, 2] = 8 subsequences
# [3, 2] not a subsequence

def seq(arr, idx, result):
    if idx == len(arr):
        print(result)
        return
    
    result.append(arr[idx])
    seq(arr, idx + 1, result)
    result.pop()
    seq(arr, idx+1, result)

arr = [3, 1, 2]
seq(arr, 0, [])