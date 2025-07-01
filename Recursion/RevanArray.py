# #  Two pointers
# def rev(a, l, r):
#     if l>=r:
#         return
#     a[l], a[r] = a[r], a[l]
#     rev(a, l+1, r-1)
# a = [1, 2, 3, 4, 5]
# n= len(a)
# rev(a, 0, n-1)
# print(a)

# One pointers
def rev(a, i, n):
    if i >= n/2:
        return
    a[i], a[n-i-1] = a[n-i-1], a[i]
    rev(a, i+1, n)

a = [1, 2, 3, 4, 5]
n= len(a)
rev(a, 0, n)
print(a)