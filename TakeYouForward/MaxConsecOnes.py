# Using XOR is the best solution to implement
#  Since they asked consecutive elements in binary array so XOR is best approach


# def MaxOnesZeros(arr):
#     cnt = 1
#     max_cnt = 0

#     for i in range(1, len(arr)):
#         if arr[i] ==arr[i-1]:
#             cnt += 1
#         else:
#             max_cnt = max(max_cnt, cnt)
#             cnt = 1

#     return max(max_cnt, cnt)

# def MaxOnesZeros(arr):
#     max_cnt, cnt, prev = 0,0,-1

#     for num in arr:
#         # check if -1^1 == 0 because 00 = 0, 11 = 0 in XOR, we check if both prev and num have equal value 
#         if (prev ^ num) == 0: 
#             cnt += 1
#         else:
#             max_cnt = max(max_cnt, cnt)
#             cnt = 1
#         prev = num 
# # if you dont write prev = num, as prev is always initialized to -1, num is XORed with -1 rather than prev value in the array 
#     return max(max_cnt, cnt)

def MaxOnesZeros(arr):
    cnt = 0
    maxi = 0
    for i in range(0, len(arr)):
        if arr[i] == 1:
            cnt += 1 
            maxi = max(cnt, maxi)
        else:
            cnt = 0
    return maxi

if __name__ == "__main__":

    arr = [1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1]
    print(MaxOnesZeros(arr))