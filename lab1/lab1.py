import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[2, 6], [4, 2], [7, 3], [9, 2]])    # берется с рисунка
y_train = np.array([1,-1,1,-1])                         # берется с рисунка
w = np.array([6, 0])                                 # берется с рисунка [b, k] по формуле прямой (y = kx + b)
n_train = len(x_train)                                  # размер обучающей выборки


print(x_train)

M = [ y * np.dot(w, x) for x, y in zip(x_train, y_train)]
print(f"M is {M}")

x_0 = x_train[y_train == 1]                 # формирование точек для 1-го
x_1 = x_train[y_train == -1]                # и 2-го классов

line_x = list(range(0,11))    # формирование графика разделяющей линии
line_y = [x*w[1] + w[0] for x in line_x]

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_y, line_x, color='green')

plt.xlim([0, 10])
plt.ylim([0, 10])
plt.ylabel('x2')
plt.xlabel('x1')
plt.grid(True)
plt.show()