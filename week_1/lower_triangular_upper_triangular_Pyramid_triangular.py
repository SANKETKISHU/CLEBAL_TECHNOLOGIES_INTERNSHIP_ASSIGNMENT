def lower_triangular(n):
    for i in range(n):
        for j in range(i + 1):
            print('*', end=' ')
        print()

def upper_triangular(n):
    for i in range(n):
        for j in range(n):
            if j >= i:
                print('*', end=' ')
            else:
                print(' ', end=' ')
        print()

def pyramid(n):
    for i in range(n):
        for j in range(n - i - 1):
            print(' ', end=' ')
        for j in range(2 * i + 1):
            print('*', end=' ')
        print()

# Example usage:
n = 5
print("Lower Triangular:")
lower_triangular(n)
print("\nUpper Triangular:")
upper_triangular(n)
print("\nPyramid:")
pyramid(n)
