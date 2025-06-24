# def removeDupes(arr):
#     index = 0
#     # Use set() to remove duplicates and add only unique numbers
#     temp = set()

#     for i in range(len(arr)):
#         if arr[i] not in temp:
#             temp.add(arr[i])
#             arr[index] = arr[i]
#             # print(arr[index], index, temp)
#             index += 1
            
#     return index 
#     # return temp

# Brute force approach
# def removeDupes(arr):
#     UniqueSet = set()
#     for num in arr:
#         UniqueSet.add(num)

#     idx=0
#     for num in UniqueSet:
#         arr[idx] = num
#         idx += 1
#     return idx

# optimal approach [two-point approach]
def removeDupes(a):
    i=0
    n=len(a)
    for j in range(1, n, 1):
        if a[j] !=a[i]:
            # a[i+1] = a[j]
            i +=1
            a[i] = a[j]

            
    return i+1

if __name__ == "__main__":
    a = [1, 2, 2, 3, 5, 5, 6, 4, 7]
    new_arr = removeDupes(a)
    print(new_arr)
    # for i in range(new_arr):
    #     print(a[i], end = " ")