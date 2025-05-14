import numpy as np
from scipy.optimize import least_squares
from matplotlib import pyplot as plt
import kagglehub
from kagglehub import KaggleDatasetAdapter
import time



cars = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mohammedadham45/cars-data",
    "cars_data.csv",
)


# Conta por ano todos os carros com trnasmissão automática
df = cars.query('transmission == "Automatic" and year < 2020')[['price', 'year']].groupby('year').mean()
print(df)

plt.plot(df.index, df['price'], c='b', marker='o')
plt.xlabel('Ano')
plt.ylabel('Valor médio')
plt.title('Valor médio de carros automáticos por ano')
plt.legend()
plt.savefig('valor_medio.png')
plt.show()


# N polynomial function
# def model(beta, x):
#     n = len(beta)
#     result = 0
#     for i in range(n):
#         result += beta[i] * x ** (n - i - 1)
#     return result


# # Modelo não linear Polinomial Grau 5
def model(beta, x):
    return beta[0] * x**5 + beta[1] * x ** 4 + beta[2] * x ** 3 + beta[3] * x ** 2 + beta[4] * x + beta[5]
# # Modelo não linear Polinomial Grau 4
# def model(beta, x):
#     return beta[0] * x**4 + beta[1] * x ** 3 + beta[2] * x ** 2 + beta[3] * x + beta[4]

# # # Modelo  Grau 3
# def model(beta, x):
#     return beta[0] * x**3 + beta[1] * x ** 2 + beta[2] * x + beta[3]

# # # Modelo Grau 2
# def model(beta, x):
#     return beta[0] * x**2 + beta[1] * x + beta[2]

# # # Modelo Grau Linear
# def model(beta, x):
#     return beta[0] * x + beta[1]

# # # Modelo não linear Logístico
# def model(beta, x):
#     return beta[0] / (1 + beta[1] * np.exp(x - beta[2]))

# Chute inicial para os parâmetros
beta0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
# beta0 = np.array([1.0, 1.0, 1.0])
# beta0 = np.array([1.0, 1.0])
# beta0 = np.array([1.0, 1.0])


y_values = []
for i in df.values.tolist():
    y_values.append(i[0])

x_data = np.array(df.index.to_list())
y_data = np.array(y_values)

# Função de resíduos
def residuals(beta, x, y):
    return y - model(beta, x)

# Ajuste usando Levenberg-Marquardt
result = least_squares(residuals, beta0, args=(x_data, y_data), method='lm')

# Plotando a curva ajustada
x_fit = np.linspace(1970, 2024, 100)

y_fit = model(result.x, x_fit)

plt.plot(x_fit, y_fit, 'r-', label='Ajuste')
plt.plot(x_data, y_data, c='b', marker='o')
plt.xlabel('X Data')
plt.ylabel('Y Data')
plt.title('Ajuste do Modelo Linear')
plt.legend()
plt.show()

plt.savefig('modelo_ajustado.png')

print("Parâmetros ajustados:", result.x)
print("Função de custo:", result.cost)


# Calcular valor médio de carros automáticos para 2020
df_2020 = cars.query('transmission == "Automatic" and year == 2020')[['price', 'year']].groupby('year').mean()
print("Valor médio de carros automáticos para 2020:", df_2020['price'].values[0])
# Calcular valor médio de carros automáticos para 2020
print("Valor calculado para 2020:", model(result.x, 2020))
print("Valor residual para 2020:", model(result.x, 2020) - df_2020['price'].values[0])
