# # brute force approach : this takes O(2(n+m)) ---TC, O(n+m)---SC
# def mergeSortArr(ar1, ar2, n, m):
#     ar3 = [0] *(n + m)
#     left = 0 
#     right = 0
#     idx = 0

#     while left < n and right < m:
#         if ar1[left] <= ar2[right]:
#             ar3[idx] = ar1[left]
#             idx += 1
#             left += 1
#         else:
#             ar3[idx] = ar2[right]
#             idx += 1
#             right += 1
#     while left < n:
#         ar3[idx] = ar1[left]
#         left += 1
#         idx += 1
#     while right < m:
#         ar3[idx] = ar2[right]
#         right += 1
#         idx += 1
#     for i in range(n + m):
#         if i < n:
#             arr1[i] = ar3[i]
#         else:
#             arr2[i - n] = ar3[i] 
#     return ar3  

# # optimal soln
def mergeSortArr(ar1, ar2, n, m):
    
    left = n - 1
    right = 0

    # Swap the elements until arr1[left] is smaller than arr2[right]:
    while left >= 0 and right < m:
        if arr1[left] > arr2[right]:
            arr1[left], arr2[right] = arr2[right], arr1[left]
            left -= 1
            right += 1
        else:
            break

    # Sort arr1[] and arr2[] individually:
    ar1.sort()
    ar2.sort()
    return ar1 +ar2

# Optimal soln : Gap method ---> from shell sort
# def mergeSortArr(ar1, ar2, n, m):

# either of the optimal soln can be told in the interview




if __name__ == '__main__':
    arr1 = [1, 4, 8, 10]
    arr2 = [2, 3, 9]
    n = 4
    m = 3
    print(mergeSortArr(arr1, arr2, n, m))