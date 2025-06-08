# Second largest element in the array
# Plan: use sorting to rearrange the elements, traverse through the loop, access the second largest with its index position

def SecondLargest(arr):
    n=len(arr)
    arr.sort()

    for i in range(n):
        if arr[i]!= arr[n-1]:
            return arr[-2]
    return -1

if __name__ == "__main__":
    # arr = [12,35,1,10,34,1]
    arr = [10, 20, 80]
    print(SecondLargest(arr))
