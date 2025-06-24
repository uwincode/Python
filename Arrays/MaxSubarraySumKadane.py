def MaxSubarray(arr):
    maxi = 0
    sum = 0
    n = len(arr)
    for i in range(n):
        sum += arr[i]
        if sum > maxi:
            maxi = sum
        elif sum < 0:
            sum = 0
    return maxi

arr = [-2, -3, 4, -1, -2, 1, 5, -3]
print(MaxSubarray(arr))