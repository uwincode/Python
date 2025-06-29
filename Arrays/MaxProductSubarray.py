# Brute force approach
# def maxProdSubarray(arr):
#     res = float('-inf')
#     n = len(arr)
#     for i in range(n-1):
#         for j in range(i+1, n):
#             prod = 1
#             for k in range(i, j+1):
#                 prod *= arr[k]
#             res = max(prod, res)
#     return res

# Better approach
def maxProdSubarray(arr):
    res = float('-inf')
    n = len(arr)
    for i in range(n-1):
        prod = 1
        for j in range(i+1, n):
            prod *= arr[j]
            res = max(prod, res)
    return res


# Optimal soln
# def maxProdSubarray(arr):
#     pre = 1
#     suff = 1
#     ans = 0
#     n = len(arr)
#     for i in range(n):
#         if pre ==0:
#             pre = 1
#         if suff == 0:
#             suff = 1
#         pre *= arr[i]
#         suff *= arr[n-i-1]
#         ans = max(ans, max(pre,suff))
#     return ans

arr = [1, 2, -3, 0, -4, -5]
print("The maximum product subarray is:", maxProdSubarray(arr))