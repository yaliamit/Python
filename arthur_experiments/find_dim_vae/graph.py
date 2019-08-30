import numpy as np
import matplotlib.pyplot as plt

# y = np.loadtxt("results_best.txt")
# x = [16, 20, 24, 28, 32, 48, 64]
# plt.plot(x, y, label="VAE")

# y = [110, 108, 105, 103, 105, 102]
# x = [8, 20, 30, 32, 48, 64]
# plt.plot(x, y, "--", label="TVAE-Aff")

# y = [97, 92, 92, 92.5, 92.8]
# x = [20, 30, 32, 48, 64]
# plt.plot(x, y, "-.", label="TVAE-TPS")

# y = [111, 95, 92]
# x = [8, 20, 30]
# plt.plot(x, y, ":", label="DTVAE-Aff")

x = [2, 10, 20, 40, 60, 80]
y = [132.16, 88.71, 88.33, 93.03, 89.76, 87.73]
x = np.array(x)
plt.plot(x, y, label="VAE", c='r')

y = [103.80, 92.6, 91.70, 90.25, 91.10, 89.74]
plt.plot(x + 6 , y, label="TVAE-Aff", c='b')
# plt.plot(x, y, '--', c='b')

y = [88.53, 83.77, 82.11, 83.36, 83.56, 82.26]
plt.plot(x + 18, y, label="TVAE-TPS", c='g')
# plt.plot(x, y, '--', c='g')

y = [102.65, 86.59, 84.92, 81.88, 83.80, 83.74]
plt.plot(x + 6, y, label="DTVAE-Aff", c='c')
# plt.plot(x, y, '--', c='c')

# y = [111, 95, 92]
# x = [8, 20, 30]
# plt.plot(x, y, ":", label="DTVAE-Aff")

plt.legend()
plt.xlabel("Number of latent variables")
plt.ylabel("Reconstruction error")
plt.show()
