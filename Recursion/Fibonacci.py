def fib(n):
    if n <= 1:
        return n
    last = fib(n-1)
    sec_last = fib(n-2)
    return last + sec_last

print(fib(4))

# tc =  ~2^n