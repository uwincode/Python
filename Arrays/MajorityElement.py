# brute force
def majorityEle(arr):
    n = len(arr)
    
    for i in range(n):
        cnt = 0
        for j in range(n):
            if arr[i] == arr[j]:
                cnt +=1 

        if cnt > n//2:
            return arr[i]
            
    return -1

# better would be hashing

# optimal soln - Moore's voting algorithm
# two variables - element, count
# check the count of each element:if it is greater than n//2 else count--
# the majority element will not cancel and remains till end
def majorityEle(arr):
    cnt = 0
    ele = 0
    n = len(arr)
    for i in range(n):
        if (cnt == 0):
            cnt = 1
            ele = arr[i]
        elif arr[i] == ele:
            cnt += 1
        else:
            cnt -= 1
    # cnt1 = 0
    # for i in range(n):
    #     if arr[i] == ele:
    #         cnt+=1
    # if cnt1 > n//2:
    #     return ele
        return ele
              

if __name__ == "__main__":
	arr = [1, 3, 3, 2, 1, 3, 5, 5]
	print(majorityEle(arr))

