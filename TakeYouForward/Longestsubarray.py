def LongestSubarray(arr, k):
    left = 0
    right = 0
    maxlen = 0
    n = len(arr)
    sum = 0
    while right<n:
        sum += arr[right]
        while left <= right and sum > k:
            sum -= arr[left]
            left+=1
        if sum==k:
            maxlen = max(maxlen, right - left +1)
        right +=1
        
            
    return maxlen

arr = [1, 2, 3, 1, 1, 1, 1]
k = 5
print(LongestSubarray(arr, k))

# there is something wrong here giving wrong output - need to check again