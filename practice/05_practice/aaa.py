import time
import math
import sys
import numpy as np

# 1

def loop_sum(a):
    total = 0.0
    for x in a:
        total += x
    return total


def numpy_sum(a):
    return a.sum()

def measure(func, array, repeat=3):
    best_time = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = func(array)
        t1 = time.perf_counter()
        best_time = min(best_time, t1 - t0)
    return result, best_time

print(" № 1")

for n in [10**5, 10**6, 10**7]:

    a = np.random.default_rng(42).normal(size=n)

    r_loop, t_loop = measure(loop_sum, a, repeat=3)

    r_np, t_np = measure(numpy_sum, a, repeat=3)

    print(f"n = {n}")
    print(f"   цикл:  {t_loop * 1000:10.2f} мс")
    print(f"   NumPy: {t_np * 1000:10.2f} мс")
    print(f"   ускорение: {t_loop / t_np:.1f}x")
    print(f"   совпадает: {np.isclose(r_loop, r_np)}")
    print("-" * 30)

# 2

def loop_mean_std(a):
    n = len(a)
    s = 0.0
    s2 = 0.0
    for x in a:
        s += x
        s2 += x * x
    mean = s / n
    var = s2 / n - mean * mean
    if var < 0:
        var = 0.0
    return mean, math.sqrt(var)


def numpy_mean_std(a):
    return a.mean(), a.std()


print("\n № 2")

a = np.random.default_rng(42).normal(size=10**6)

(m_loop, s_loop), t_loop = measure(loop_mean_std, a)
(m_np, s_np), t_np = measure(numpy_mean_std, a)

print(f"mean: цикл={m_loop:.15f}  numpy={m_np:.15f}  разница={abs(m_loop - m_np):.2e}")
print(f"std:  цикл={s_loop:.15f}  numpy={s_np:.15f}  разница={abs(s_loop - s_np):.2e}")
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")

ok_mean = abs(m_loop - m_np) < 1e-12
ok_std  = abs(s_loop - s_np) < 1e-12
print(f"  |Δmean| = {abs(m_loop - m_np):.2e},  < 1e-12? {ok_mean}")
print(f"  |Δstd|  = {abs(s_loop - s_np):.2e},  < 1e-12? {ok_std}")
print(f"  совпадает до 1e-12: {ok_mean and ok_std}")

# 3

def loop_transform(a):
    result = []
    for x in a:
        result.append(math.sin(x) + x * x)
    return result


def numpy_transform(a):
    return np.sin(a) + a**2


print("\n № 3")

a = np.random.default_rng(42).normal(size=10**6)

r_loop, t_loop = measure(loop_transform, a)
r_np, t_np = measure(numpy_transform, a)

mem_list = len(r_loop) * 32 
mem_np   = r_np.nbytes
print(f"  память: список ~{mem_list/1e6:.1f} МБ, ndarray {mem_np/1e6:.1f} МБ "
      f"(в {mem_list/mem_np:.1f}× компактнее)")

print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")

# 4

def loop_clip(a):
    result = []
    for x in a:
        if x < 0:
            result.append(0.0)
        elif x > 1:
            result.append(1.0)
        else:
            result.append(x)
    return result


def numpy_clip(a):
    return np.clip(a, 0, 1)


print("\n № 4")

a = np.random.default_rng(42).normal(size=10**6)

r_loop, t_loop = measure(loop_clip, a)
r_np, t_np = measure(numpy_clip, a)

print(f"  изменено элементов: {int(np.sum((a < 0) | (a > 1)))}")
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")

# 5

def loop_sum_rows(m):
    n, k = m.shape
    result = [0.0] * n
    for i in range(n):
        s = 0.0
        for j in range(k):
            s += m[i, j]
        result[i] = s
    return result


def numpy_sum_rows(m):
    return m.sum(axis=1)


print("\n № 5")

m = np.random.default_rng(42).normal(size=(10**5, 10))

r_loop, t_loop = measure(loop_sum_rows, m)
r_np, t_np = measure(numpy_sum_rows, m)

print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")


# 6

def loop_center(m):
    n, k = m.shape
    means = [0.0] * k
    for j in range(k):
        s = 0.0
        for i in range(n):
            s += m[i, j]
        means[j] = s / n

    result = [[0.0] * k for _ in range(n)]
    for i in range(n):
        for j in range(k):
            result[i][j] = m[i, j] - means[j]
    return result


def numpy_center(m):
    return m - m.mean(axis=0)


print("\n № 6")

m = np.random.default_rng(42).normal(size=(10**5, 10))

t0 = time.perf_counter()
r_loop = loop_center(m)
t1 = time.perf_counter()
t_loop = t1 - t0

t0 = time.perf_counter()
r_np = numpy_center(m)
t1 = time.perf_counter()
t_np = t1 - t0

print(f"  средние после нормировки: {np.round(r_np.mean(axis=0), 12)}")
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")

# 7

def loop_max(a):
    max_val = a[0]
    max_idx = 0
    for i in range(len(a)):
        if a[i] > max_val:
            max_val = a[i]
            max_idx = i
    return max_val, max_idx


def numpy_max(a):
    return a.max(), a.argmax()


print("\n № 7")

a = np.random.default_rng(42).normal(size=10**6)

(v_loop, i_loop), t_loop = measure(loop_max, a)
(v_np, i_np), t_np = measure(numpy_max, a)

print(f"  max: цикл={v_loop:.6f} idx={i_loop}  numpy={v_np:.6f} idx={i_np}")
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.isclose(v_loop, v_np) and i_loop == i_np}")

a_nan = a.copy()
a_nan[0] = np.nan
print(f"  с NaN: цикл даёт {loop_max(a_nan)[0]}, np.max даёт {a_nan.max()}, np.nanmax даёт {np.nanmax(a_nan):.6f}")

# 8

def loop_filter(a):
    result = []
    for x in a:
        if x > 0:
            result.append(x)
    return result


def numpy_filter(a):
    return a[a > 0]


print("\n № 8")

a = np.random.default_rng(42).normal(size=10**6)

r_loop, t_loop = measure(loop_filter, a)
r_np, t_np = measure(numpy_filter, a)

n_kept = len(r_loop)
mem_list = n_kept * 32
mem_np   = r_np.nbytes

print(f"  память: список ~{mem_list/1e6:.1f} МБ, ndarray {r_np.nbytes/1e6:.1f} МБ")
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")

# 9

def loop_abs_inplace(a):
    for i in range(len(a)):
        if a[i] < 0:
            a[i] = -a[i]


def numpy_abs_inplace(a):
    a[a < 0] *= -1


print("\n № 9")

a1 = np.random.default_rng(42).normal(size=10**6)
a2 = a1.copy()

def measure_inplace(func, array, repeat=3):
    best_time = float("inf")
    for _ in range(repeat):
        a_copy = array.copy()
        t0 = time.perf_counter()
        func(a_copy)
        t1 = time.perf_counter()
        best_time = min(best_time, t1 - t0)
    return best_time


t_loop = measure_inplace(loop_abs_inplace, a1)
t_np = measure_inplace(numpy_abs_inplace, a2)

loop_abs_inplace(a1)
numpy_abs_inplace(a2)


print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(a1, a2)}")

# 10

def loop_mask(a):
    result = []
    for x in a:
        if -1 < x < 1:
            result.append(x)
    return result


def numpy_mask(a):
    return a[(a > -1) & (a < 1)]


print("\n № 10")

a = np.random.default_rng(42).normal(size=10**6)

r_loop, t_loop = measure(loop_mask, a)
r_np, t_np = measure(numpy_mask, a)

print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")

# 11

def loop_dist(X, Y):
    n = len(X)
    m = len(Y)
    D = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(3):
                d = X[i, k] - Y[j, k]
                s += d * d
            D[i][j] = math.sqrt(s)
    return D


def numpy_dist(X, Y):
    return np.sqrt(((X[:, None, :] - Y[None, :, :])**2).sum(axis=2))


print("\n № 11")

X = np.random.default_rng(42).normal(size=(1000, 3))
Y = np.random.default_rng(43).normal(size=(1000, 3))

r_loop, t_loop = measure(lambda arr: loop_dist(X, Y), None)
r_np, t_np = measure(lambda arr: numpy_dist(X, Y), None)

print(f"  временный массив NumPy: {1000*1000*3*8/1e6:.0f} МБ")
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")

# 12

def loop_moving_avg(a, k=10):
    n = len(a)
    result = []
    for i in range(k - 1, n):
        s = 0.0
        for j in range(i - k + 1, i + 1):
            s += a[j]
        result.append(s / k)
    return result


def numpy_moving_avg(a, k=10):
    return np.convolve(a, np.ones(k) / k, mode="valid")


print("\n № 12")

a = np.random.default_rng(42).normal(size=10**5)

r_loop, t_loop = measure(loop_moving_avg, a)
r_np, t_np = measure(numpy_moving_avg, a)

print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.allclose(r_loop, r_np)}")

# 13

def loop_cumsum(a):
    result = []
    s = 0.0
    for x in a:
        s += x
        result.append(s)
    return result


def numpy_cumsum(a):
    return np.cumsum(a)


print("\n № 13")

a = np.random.default_rng(42).normal(size=10**5)

r_loop, t_loop = measure(loop_cumsum, a)
r_np, t_np = measure(numpy_cumsum, a)

diff = float(np.max(np.abs(np.array(r_loop) - r_np)))
print(f"  max разница: {diff:.2e}")
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает до 1e-9: {np.allclose(r_loop, r_np, atol=1e-9)}")


# 14

def loop_bincount(ints):
    counts = {}
    for v in ints:
        counts[v] = counts.get(v, 0) + 1
    return counts


def numpy_bincount(ints):
    return np.bincount(ints, minlength=100)

print("\n № 14")

ints = np.random.default_rng(42).integers(0, 100, size=10**6)

r_loop, t_loop = measure(loop_bincount, ints)
r_np, t_np = measure(numpy_bincount, ints)

loop_arr = np.array([r_loop.get(i, 0) for i in range(100)])
print(f"  цикл:  {t_loop*1000:10.2f} мс")
print(f"  NumPy: {t_np*1000:10.2f} мс")
print(f"  ускорение: {t_loop / t_np:.1f}x")
print(f"  совпадает: {np.array_equal(loop_arr, r_np)}")