# def triplet(n, arr):
#     st = set()
#     for i in range(n):
#         for j in range(i+1, n):
#             for k in range(j+1, n):
#                 if arr[i] + arr[j] + arr[k] == 0:
#                     temp = [arr[i], arr[j], arr[k]]
#                     # first sort the temp 
#                     temp.sort()
#                     # then add it in set to avoid the duplicates
#                     st.add(tuple(temp))  #if you remove tuple here, you will see the error
    
    # return st

# if you took 2, -1 as i,j elements, and check -(2-4) = 2 in the array then its wrong that's why you use hashing
# def triplet(n, arr):
#     st = set()
#     for i in range(n):
#         hashset = set() #everytime i is incremented, hashset is reinitialized
#         for j in range(i+1, n):
#             third = - (arr[i] + arr[j])
#             if third in hashset:
#                 temp = [arr[i], arr[j], third]
# #                     # first sort the temp 
#                 temp.sort()
#                 st.add(tuple(temp))
#             hashset.add(arr[j])  #everytime j moves or inc, add that j into set
#     ans = list(st)
#     return ans

# optimal soln: i is constant, j and k will traverse
def triplet(n, arr):
    res =[]
    arr.sort()
    for i in range(n):
        if arr[i] == arr[i-1] and i !=0:
            continue
        j =i+1
        k = n-1
        while j < k:
            sum = arr[i] + arr[j] +arr[k]
            if sum < 0 : #since elements are sorted, sum < 0  (1st ele of lower value)
                # then inc j or sum> 0 then dec k (last ele will be of higher value)
                j += 1
            elif sum> 0:
                k -= 1
            else:
                temp = [arr[i], arr[j], arr[k]]
                res.append(temp)
                j+=1
                k-=1
                # skip duplies
                while j < k and arr[j] == arr[j - 1]:
                    j += 1
                while j < k and arr[k] == arr[k + 1]:
                    k -= 1

    return res






if __name__ == "__main__":
    arr = [-1, 0, 1, 2, -1, -4]
    n = len(arr)
    ans = triplet(n, arr)
    print(ans)