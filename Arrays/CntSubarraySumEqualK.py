# Brute force Approach
# def CntSubarray(arr, m):
#     n = len(arr)
#     cnt = 0
#     for i in range(n):
        
#         for j in range(i, n):
#             sum=0
#             for k in range(i, j+1):   #i to j ----to include j we write j+1
#                 sum += arr[k]
#             if (sum == m):
#                 cnt+=1
#     return cnt
                
# BETTER APPROACH
# def CntSubarray(arr, k):
#     n = len(arr)
#     cnt = 0
#     for i in range(n):
#         sum=0
#         for j in range(i, n):
#             sum += arr[j]
#             if (sum == k):
#                 cnt+=1
#     return cnt

# Optimal soln
def CntSubarray(arr, m):
    n = len(arr)
    prefix_Sum = 0
    cnt = 0
    prefix_Sum_cnt = {0: 1}
    for i in range(n):
        prefix_Sum += arr[i]
        remove = prefix_Sum - m
        if remove in prefix_Sum_cnt:
            cnt += prefix_Sum_cnt[remove]

        if prefix_Sum in prefix_Sum_cnt:
            prefix_Sum_cnt[prefix_Sum] += 1
        else:
            prefix_Sum_cnt[prefix_Sum] = 1

    return cnt
        

arr = [1, 2, 3, -3, 1, 1, 1, 1, 4, 2, -3]
target = 3
print(CntSubarray(arr, target))