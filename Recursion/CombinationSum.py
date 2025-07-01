def combSum(arr, idx, target, res, path):
    if target == 0:
        res.append(path[:])  # Save a copy of the valid combination
        return
    if idx == len(arr) or target < 0:
        return
    if arr[idx] <= target:
        path.append(arr[idx])
        combSum(arr, idx, target - arr[idx], res, path)
        path.pop()
    combSum(arr, idx+1, target, res, path)

arr = [2, 3, 6, 7]
target = 7
res = []
combSum(arr, 0, target, res, [])
print(res)