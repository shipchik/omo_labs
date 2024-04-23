import numpy as np
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def f(x):
    return 0.1*x**5 - 100*x**3 + 700*x**2


x = np.arange(0, 10, 0.1)
y = f(x)


poly = PolynomialFeatures(degree=13)
X = poly.fit_transform(x.reshape(-1, 1))

# Обучение моделей с регуляризацией L1 и L2
lasso = Lasso()
ridge = Ridge()
lasso.fit(X, y)
ridge.fit(X, y)

# Предсказание значений
y_pred_lasso = lasso.predict(X)
y_pred_ridge = ridge.predict(X)

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Исходная функция')
plt.plot(x, y_pred_lasso, label='Аппроксимация с L1 регуляризацией')
plt.plot(x, y_pred_ridge, label='Аппроксимация с L2 регуляризацией')
plt.legend()
plt.show()
