def CheckSort(arr):
    n=len(arr)

    for i in range(1, n, 1):
        if arr[i] < arr[i-1]:
            return False
    return True



if __name__ == "__main__":
    # arr = [12,35,1,10,34,1]
    arr = [10, 20, 30, 80]
    print(CheckSort(arr)) 