# brute force
# def Longest(arr):
#     longest = 1
    
#     n = len(arr)
#     for i in range(n):
#         cnt = 1
#         x = arr[i]
#         for j in range(i+1, n):
#             if arr[j] == x+1 :
#                 x += 1
#                 cnt += 1
#         longest = max(cnt, longest)
#     return longest

# Better soln can be done using sort, and then check consec elements with its last smaller
# check in youtube video 

# Optimal Soln
# the first element will not have previous
#  first put all elements in set --> to avoid duplicates
# check if the element has prev, or next 

def Longest(arr):
    longest = 1
    n = len(arr)
    arr_set = set(arr)

    for num in arr_set:
        if num-1 not in arr_set:
            curr = num
            cnt = 1
        
        while curr + 1 in arr_set:
            curr += 1
            cnt += 1
        
        longest = max(longest, cnt)
    return longest
        

arr = [102, 4, 100, 1, 101, 3, 2, 1, 1]
print(Longest(arr))
                
            
