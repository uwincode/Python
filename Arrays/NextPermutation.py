# brute force soln
# Generate all permutations, sort them in dictionary order, through linear search check where the arr[] comes, print its next permutation


# Optimal soln
# 1. write longer prefix match check condn: a[i] < a[i+1] ---> breakpoint
# in that way you will break the array into two left [i], right [i+1]
# 2. find someone slightly greater than i
# 3. try to place the array in sorted order after i 

def rev(arr, start, end):
    while(start < end):
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1

def nextPerm(arr):
    n=len(arr)
    idx = -1
    for i in range(n-2, -1, -1):
        if arr[i] < arr[i+1]:
            idx = i
            break

    if idx == -1:
        rev(arr, 0, n-1)
        return arr
    
    for i in range(n-1, idx, -1):
        if arr[i] > arr[idx]:
            arr[i], arr[idx] = arr[idx], arr[i]
            break
    rev(arr, idx+1, n-1)
    return arr

arr = [2, 1, 5, 4, 3, 0, 0]
print(nextPerm(arr))

