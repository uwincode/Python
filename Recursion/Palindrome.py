def palindrome(s, i):
    if i >= n//2:
        return True
    if (s[i] != s[n-i-1]):
        return False
    return palindrome(s, i+1)

s = "madam"
n = len(s)
print(palindrome(s, 0))   