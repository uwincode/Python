# # Base Case: If n <= 0, the function returns and recursion stops.

# # Recursive Case: It prints the name once and calls itself with n - 1.
# def print_name(name, n):
#     if n<= 0:
#         return 
#     print(name)
#     print_name(name, n-1)
# print_name("Lahiri", 5)

# def name(i, n):
#     if i>n:
#         return
#     print("Lah")
#     name(i+1, n)
# name(1, 3)

# # print numbers from n to 1
# def number(i,n):
#     if i>n:  #base condition
#         return
#     print(i)
#     number(i+1, n)  #recusive
# number(1, 5)

# # print numbers from i to n w/o using recursive i+1
# def number(i, n):
#     if i <1:
#         return
    
#     number(i-1, n)
#     print(i)
# number(5, 5)

# # print numbers from n to 1 w/o using recursive i-1
# def number(i,n):
#     if i>n:
#         return
#     number(i+1, n)
#     print(i)
# number(1,5)

# # sum of n natural numbers - Parameterized
# def T_sum(i, sum):
#     if i < 1:
#         print(sum)
#         return
#     T_sum(i-1, sum+i)

# n = 5
# T_sum(n, 0)

# # sum of n natural numbers - Functional
# def func(n):
#     if n==0:
#         return 0   #if written just "return", gives type error
#     return n+func(n-1)
# print(func(3))

# factorial -- functional
def fact(n):
    if n==0:
        return 1  #prod
    return n * fact(n-1)
print(fact(4))