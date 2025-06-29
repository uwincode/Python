def mergesort(arr, low, high):
    
    # low = 0
    # high = n-1
    
    if low >= high:
        return 
    mid = (low + high)//2
    mergesort(arr, low, mid)
    mergesort(arr, mid+1, high)
    merge(arr, low, mid, high)

def merge(arr, low, mid, high):
    temp = []
    
    left = low
    right = mid + 1

    while left <= mid and right <= high:
        if arr[left] <= arr[right]:
            temp.append(arr[left])
            left +=1
        else:
            temp.append(arr[right])
            right += 1
        
      # if elements on the left half are still left
    while left <= mid:
        temp.append(arr[left])
        left += 1

    # if elements on the right half are still left
    while right <= high:
        temp.append(arr[right])
        right += 1
    # transferring all elements from temp to original array
    for i in range(low, high + 1):
        arr[i] = temp[i - low]



arr = [9, 4, 7, 6, 3, 1, 5]
mergesort(arr, 0, len(arr) - 1)
print("Sorted array:", arr)