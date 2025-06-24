# Brute Force algorithm: 
# 1. You will sort the array of elements 
# 2. Access the last element

# sort() will give you the answer in ascending order
# to access the last element you will use arr[n-1]

# time complexity: since sort is used it will be O(nlogn)
# space complexity: O(1))

# Optimal Solution:
# assume first element as largest
# traverse through the array

def LargestElement(arr):
    n=len(arr)
    largest = arr[0]
    for i in range(n):
        if (arr[i] > largest):
            largest = arr[i]
    return largest
    
if __name__ == "__main__":
    # arr = [12,35,1,10,34,1]
    arr = [10, 20, 30, 80]
    print(LargestElement(arr))


    # time complexity: O(n)
    # space complexity: O(1)