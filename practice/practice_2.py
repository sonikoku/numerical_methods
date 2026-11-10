import math
import mpmath

def experiment(f_bad, f_good, xs, f_exact):
    print(f"{'x':>10} | {'bad':>14} | {'good':>14} | "
          f"{'err_bad':>10} | {'err_good':>10}")
    print("-" * 78)
    for x in xs:
        b  = f_bad(x)
        g  = f_good(x)
        ex = f_exact(x)
        eb = abs(b - ex) / abs(ex) if ex else float('inf')
        eg = abs(g - ex) / abs(ex) if ex else float('inf')
        print(f"{x:>10.0e} | {b:>14.8e} | {g:>14.8e} | "
              f"{eb:>10.2e} | {eg:>10.2e}")


xs_small = [1e-10, 1e-6, 1e-2]
xs_big = [1e2, 1e6, 1e10, 1e14]

print("\n--- Таблица 1: Маленькие x ---")
experiment(
    f_bad   = lambda x: math.sqrt(x + 1) - math.sqrt(x),
    f_good  = lambda x: 1.0 / (math.sqrt(x + 1) + math.sqrt(x)),
    xs      = xs_small,
    f_exact = lambda x: float(mpmath.sqrt(x + 1) - mpmath.sqrt(x))
)

print("\n--- Таблица 2: Большие x ---")
experiment(
    f_bad   = lambda x: math.sqrt(x + 1) - math.sqrt(x),
    f_good  = lambda x: 1.0 / (math.sqrt(x + 1) + math.sqrt(x)),
    xs      = xs_big,
    f_exact = lambda x: float(mpmath.sqrt(x + 1) - mpmath.sqrt(x))
)