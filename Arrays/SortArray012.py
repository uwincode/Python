# brute force approach
# def SortArr(arr):

#     n = len(arr)
#     cnt0 = 0
#     cnt1 = 0
#     cnt2 = 0

#     for i in range(n):
#         if arr[i]==0:
#             cnt0 += 1
#         elif arr[i] == 1:
#             cnt1 += 1
#         else:
#             cnt2 += 1
#     for i in range(cnt0):
#         arr[i] = 0
#     for i in range(cnt0, cnt0+cnt1):
#         arr[i] = 1
#     for i in range(cnt0+cnt1, n):
#         arr[i] = 2

#     return arr



# Optimal Soln - Dutch National Flag Algorithm
# it will have 3 pointers low, mid, high, you need to sort the elements in mid to high range
# [0...low-1]--->0
# [low...mid-1]---> 1
# [mid...high]---> need to sort
# [high+1...n-1]--->2
# First you will point arr[0] as mid and low then check if
# if a[mid] == 0 then swap(a[low], a[mid]), low++, mid++
# if a[mid] == 1 then mid++
# if a[mid] == 2 then swap(a[mid], a[high]), high--

def SortArr(arr):
    n = len(arr)
    low = 0
    mid = 0
    high = n-1
    while mid <= high:
        if (arr[mid] == 0):
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif (arr[mid] == 1):
            mid += 1
        elif (arr[mid] == 2):
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr

arr = [1, 2, 0, 2, 0, 1, 0, 1, 2, 1]
print(SortArr(arr))