# Brute force algorithm (above solution)
# You will sort the array of elements
# access the second element from last that is not equal to 1st largest
# start the loop from back or last
# in an array [1 2 4 5 7 7] since i know n-1 is largest start from second element from last
# time complexity is nlogn for sort and n to travel the entire array so O(nlogn+n)

# def SecondLargest(arr):
#     n=len(arr)
#     arr.sort()

#     largest = arr[n-1]
#     for i in range(n-2, -1, -1):
#         if arr[i] != largest:
#             second = arr[i]
#             return second
#         return -1

# if __name__ == "__main__":
#     # arr = [12,35,1,10,34,1]
#     arr = [10, 20, 30, 80]
#     print(SecondLargest(arr))





# # Better solution 
# time complexity: two for loop O(n) + O(n) = O(2n)

# # be in the array
# def SecondLargest(arr):
#     sl = -1
#     largest = arr[0]
#     n=len(arr)
#     for i in range(n):
#         if (arr[i] > largest):
#             largest = arr[i]
        
#     for i in range(n):
#         if (arr[i] > sl and arr[i]< largest):
#             sl = arr[i]
#     return sl   



# Optimal solution
def SecondLargest(arr):
    sl = -1
    largest = arr[0]
    n=len(arr)
    for i in range(n):
        if (arr[i]>largest):
            sl = largest
            largest = arr[i]

        elif (arr[i] > sl and arr[i]<largest):
            sl = arr[i]

    return sl


if __name__ == "__main__":
    # arr = [12,35,1,10,34,1]
    arr = [10, 20, 30, 80]
    print(SecondLargest(arr)) 