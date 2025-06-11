def RevArrByKEle(arr,k):
    i = 0
    n = len(arr)

    while i<n:
        left = i
        right = min(i+k-1, n-1)
        while left < right:
            temp = arr[left]
            arr[left] = arr[right]
            arr[right] = temp
            left += 1
            right -= 1
        i += k

arr = [1, 2, 3, 4, 5 , 6, 7, 8]
k=3
RevArrByKEle(arr, k) 

print(" ".join(map(str, arr)))