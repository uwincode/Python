def combSum(arr, idx, target, path, res):
    
    if target == 0:
            res.append(path[:])
            return
    if target <0:
         return
    
    n = len(arr)
    for i in range(idx, n): 
        # i>idx is important to check
        if i > idx and arr[i] == arr[i - 1]:
            continue  
        if arr[i] > target:
            break     # skip elements that exceed the target
        
        path.append(arr[i])
         # i + 1 because we can't reuse the same element
        combSum(arr, i+1, target - arr[i], path, res)
        path.pop()
    # combSum(arr, idx+1, target, res, path)

# arr = [2, 3, 6, 7]
arr = [1, 1, 1, 2, 2]
arr.sort()
# target = 7
target = 4
res = []
combSum(arr, 0, target, [], res)
print(res)