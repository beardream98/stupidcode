def factorial(n):
    """
    return n
    
    """

    return 1 if n<2 else n*factorial(n-1)
print(factorial(42))
print(factorial.__doc__)
print(type(factorial))

#赋值
fact=factorial
#作为参数
print(list(map(factorial,range(11))))