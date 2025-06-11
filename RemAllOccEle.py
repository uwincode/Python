def removeElement(arr, ele):
    cnt = 0
    n = len(arr)
    for i in range(n):
        # if arr[i] == ele:
            # arr.remove(ele) this give index error
        # else:
        #     cnt += 1

        if arr[i] != ele:
            
            arr[cnt] = ele
            cnt += 1
    return cnt



if __name__ == "__main__":
    arr = [0, 1, 3, 0, 2, 2, 4, 2]
    ele = 2
    print(removeElement(arr, ele))
