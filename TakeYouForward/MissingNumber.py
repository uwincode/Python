# brute force approach
# def findMissingNumber(arr):
#     n = len(arr)+1 #Include the missing number
#     for i in range(1, n, 1):   # 1,2, 3, 4, 5
#         flag = 0
#         for j in range(len(arr)): # 1, 2, 4, 5
#             if arr[j]==i:
#                 flag = 1
#                 break
#         if flag==0:
#             return i

#this part below code is wrong and with mistakes
# def findMissingNumber(arr):
#     n = len(arr)
#     hash[n+1] = [0]
#     for i in range(n):
#         if hash[arr[i]]:
#             hash[arr[i]] += 1
#     for i in range(1, n):
#         if hash[arr[i] == 0]:
#             return i    



#optimal approach
# def findMissingNumber(arr):
#     n=len(arr)+1
#     exp_sum = n * (n+1)//2
#     act_sum = 0
    
#     act_sum = sum(arr)
#     return exp_sum - act_sum

# Another optimal approach
# def findMissingNumber(arr):
#     n= len(arr)+1
#     xor1 = 0
#     xor2 = 0
#     for i in range(1, n+1): #1, 2, 3, 4, 5
#         xor1 = xor1 ^ i
#     for i in range(0, n-1): 
#         xor2 = xor2 ^ arr[i]
#     return xor1^xor2

# Best approach
def findMissingNumber(arr):
    n= len(arr)+1
    xor1 = 0
    xor2 = 0
    
    for i in range(0, n-1): 
        xor2 = xor2 ^ arr[i]  # 1, 2, 4. 5
        xor1 = xor1 ^ (i+1)   # 1-->4 (1, 2, 3, 4)

    xor1 = xor1^n #5
    return xor1^xor2



if __name__ == "__main__":
    arr = [1, 2, 4, 5, 6]  # missing 3
    missing = findMissingNumber(arr)
    print("Missing number is:", missing)


