import numpy as np

# 1.
print("\n")

bark = np.arange(24).reshape(6, 4)
print(bark.shape, bark.ndim, bark.size, bark.dtype)
print("\n")

# 2.
bark_4x6 = bark.reshape(4, 6)
bark_3d = bark.reshape(2, 3, 4)
bark_flat = bark.reshape(24)
print(np.array_equal(bark, bark_4x6.reshape(6, 4))) 
print("\n")

# 3.
column_view = bark[:, 2]
column_view[0] = 999
print(bark)
print("\n")

# 4.
mask = bark > 10
filtered = bark[mask]
print(filtered.mean())
print("\n")

# 5.
row_means = bark.mean(axis=1, keepdims=True)
centered_bark = bark - row_means
print(centered_bark)
print(centered_bark.sum(axis=1))
print("\n")

