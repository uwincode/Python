# takes an element and place it is correct order


def insertsort(arr):
    n = len(arr)
    for i in range(1, n): 
        j = i
        while (j > 0 and arr[j-1] > arr[j] ) :
        
            arr[j-1], arr[j] = arr[j], arr[j-1]
            j -= 1
    return arr

arr= [13,46,24,52,20,9]
print(insertsort(arr))
