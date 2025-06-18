# brute force approach
# def Element(arr):
#     for i in range(0,len(arr)):
#         num = arr[i]
#         cnt = 0
#         for j in range(0, len(arr)):
            
#             if arr[j] == num:
#                 cnt += 1
#         if cnt == 1 :
#             return num


# better soln can be done by hashing

# OPtimal soln
def Element(arr):
    xor = 0
    for i in range(0,len(arr)):
        xor = xor ^ arr[i]
    return xor
    
arr = [4, 3, 2, 4, 2, 5, 3]
print("First non-repeating element:", Element(arr))