def removeDupes(arr):
    index = 0
    # Use set() to remove duplicates and add only unique numbers
    temp = set()

    for i in range(len(arr)):
        if arr[i] not in temp:
            temp.add(arr[i])
            arr[index] = arr[i]
            print(arr[index], index, temp)
            index += 1
            
    return index 
    # return temp

if __name__ == "__main__":
    arr = [1, 2, 2, 3, 4, 4, 4, 5, 5]
    new_arr = removeDupes(arr)
    for i in range(new_arr):
        print(arr[i], end = " ")