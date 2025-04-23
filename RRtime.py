import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r

# Passo 1: Dados de chuvas máximas anuais
chuvas_maximas = np.array([])  # em mm

# Cálculo da média e desvio padrão
media = np.mean(chuvas_maximas)
desvio_padrao = np.std(chuvas_maximas, ddof=1)  # ddof=1 para amostra (n-1)
print(f"Média das chuvas máximas: {media:.2f} mm")
print(f"Desvio padrão das chuvas máximas: {desvio_padrao:.2f} mm\n")

# Passo 2: Ordenar (Weibull)
n = len(chuvas_maximas)
chuvas_ordenadas = np.sort(chuvas_maximas)[::-1]  # Decrescente
m = np.arange(1, n + 1)
P = m / (n + 1)  # Fórmula de Weibull
T_empirico = 1 / P  # Tempo de recorrência empírico

# Passo 3: Ajustar a distribuição de Gumbel aos dados
loc, scale = gumbel_r.fit(chuvas_maximas)
print(f"Parâmetros de Gumbel: loc = {loc:.2f}, scale = {scale:.2f}")

# Passo 4: Cálculo do Tempo de Recorrência
T_alvo = np.array([2, 5, 10, 25, 50, 100, 1000, 10000]
                  )  # Tempos de recorrência (anos)
P_alvo = 1 / T_alvo  # Probabilidades de excedência
Q_T = gumbel_r.ppf(1 - P_alvo, loc=loc, scale=scale)  # Valores estimados

# Passo 5: Gráfico dos Resultados
plt.figure(figsize=(10, 6))
plt.scatter(T_empirico, chuvas_ordenadas, color='red', label='Dados Empíricos')
plt.plot(T_alvo, Q_T, marker='o', color='blue', label='Distribuição de Gumbel')
plt.xscale('log')  # Escala logarítmica
plt.xlabel('Tempo de Recorrência (anos)')
plt.ylabel('Chuva Máxima (mm)')
plt.title('Estudo de Tempo de Recorrência de Chuvas Máximas')
plt.grid(True, which="both", ls="--")
plt.legend()
plt.show()

# Passo 6: Tabela formatada
print("\nTabela de Tempo de Recorrência vs. Chuva:")
print("----------------------------------------")
print("T (anos) | Chuva (mm)")
print("----------------------------------------")
for T, Q in zip(T_alvo, Q_T):
    print(f"{T:5}   | {Q:8.2f}")
print("----------------------------------------")
