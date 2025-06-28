# brute force
def cntRevPair(arr):
    n = len(arr)
   
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            if arr[i] > 2*arr[j]:
                cnt += 1
    return cnt
a = [4, 1, 2, 3, 1]
print(cntRevPair(a))

# optimal soln 
# Merge sort - easy to understand - need to learn how to write merge sort pgrm so i can come back and complete this