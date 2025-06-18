# brute force soln
# def TwoSum(arr, T):
#     for i in range(len(arr)):
#         for j in range(len(arr)):
#             if (arr[i] + arr[j]) == T:
#                 return [i, j]



# better soln uses hashing

# optimal soln - Two Point Array

def TwoSum(arr, T):
    i = 0
    n = len(arr)
    j = n-1
    while i<j:
        sum = arr[i] + arr[j]
        if sum == T:
            return (i, j)
        elif sum< T:
            i+=1
        else:
            j-=1
    return -1
                
arr = [2, 5, 6, 8, 11]
T = 19
print(TwoSum(arr, T))
                
                
