# push the max to the last by "adjacent" swaps
# check if 13, 46 needs swap --- yes
# check if 46, 24 needs swap --- no ---swap 13 24 46 52 20 9
# check if 46, 52 needs swap --- yes
# check if 52, 20 needs swap --- no ---swap 13 24 46 20 52 9
# check if 52, 9 needs swap --- no ---swap 13 24 46 20 9 52
# by the end of first traverse (one complete adjacent swap), the max of arr will be in the last
# so check until last but one 13 24 46 20 9 || 52

def bubblesort(arr):
    n = len(arr)
    for i in range(n-1, -1, -1): 
        for j in range(0, i): 
            # loop from 0 to i-1 --- 0 to n-2
            #in last iter, for 52, no ele to compare
            #if written j from 0 to i ---it throws out of bound index
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]

    return arr

arr= [13,46,24,52,20,9]
print(bubblesort(arr))
