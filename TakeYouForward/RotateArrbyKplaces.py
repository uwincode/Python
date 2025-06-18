# Look the take u forward rotate arrray by k places to know why we wrote d=d%n


def rotate(arr, d):
    n=len(arr)
    d%=n
    rev(arr, 0, d-1)
    rev(arr, d, n-1)
    rev(arr, 0, n-1)

def rev(arr, start, end):
    while(start < end):
        arr[start], arr[end] = arr[end], arr[start]
        start += 1
        end -= 1

if __name__ == "__main__":
    arr = [1, 2, 3, 4, 5, 6]
    d = 2
    rotate(arr, d)
    for i in range(len(arr)):
        print(arr[i], end = " ")
 