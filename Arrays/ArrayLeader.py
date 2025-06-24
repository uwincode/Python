# #  everthing to its(leader element) right should be smaller
# # brute force 
# # go across every element through linear search and check if there is any element greater than i
# def arrayLeader(arr):
#     n = len(arr)
#     res = []
#     for i in range(n):
#         leader = True
#         for j in range(i+1, n):
#             if arr[j] > arr[i]:
#                 leader = False
#                 break
#         if leader: #leader == True
#             res.append(arr[i]) 
#     return res

# Optimal soln
def arrayLeader(arr):
    res = []
    max_right = arr[-1]
    res.append(max_right)
    n = len(arr)
    for i in range(n-2, -1, -1):
        
        if arr[i] >= max_right:

            max_right = arr[i]
            res.append(max_right)

    res.reverse()
    return res

if __name__ == "__main__":
    arr = [16, 17, 4, 3, 5, 2]
    res = arrayLeader(arr)
    print(" ".join(map(str, res)))
