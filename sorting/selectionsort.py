# select the min value and then swap 

def selectsort(arr):
    n = len(arr)
    for i in range(n-1): #loop ranges from 0 to n-1 and 
        # runs from 0 to n-2(until last but one --- bcoz by last iteration, last ele would already be sorted and no need to run  )
        mini = i
        for j in range(i+1, n):
            if arr[j] < arr[mini]:
                mini = j
        arr[i], arr[mini] = arr[mini], arr[i]

    return arr

arr= [13,46,24,52,20,9]
print(selectsort(arr))