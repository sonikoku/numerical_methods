import math

def relative_error(approx, exact):
    return abs(approx - exact) / abs(exact)

# =====
print("1. Пи:")
print("Δx =", abs(3.14 - math.pi))
print("δx =", relative_error(3.14, math.pi))

# =====
print("\n2. 0.1 + 0.2:")
print("Δx =", abs((0.1 + 0.2) - 0.3))
print("δx =", relative_error(0.1 + 0.2, 0.3))

# =====
print("\n3. Сумма миллиона 0.1:")
vals = [0.1] * 1_000_000
exact = 100_000.0

total_loop = sum(vals)
print("Обычный цикл: Δx =", abs(total_loop - exact), " δx =", relative_error(total_loop, exact))

total_fsum = math.fsum(vals)
print("math.fsum:    Δx =", abs(total_fsum - exact), " δx =", relative_error(total_fsum, exact))