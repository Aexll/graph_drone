import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt

# Paramètres de la distribution bêta
alpha = 1
beta_param = 5  # Renommé pour éviter le conflit avec le module beta de scipy

# Crée la distribution bêta
dist = beta(alpha, beta_param)

# Génère des points sur l'intervalle [0, 1]
x = np.linspace(0, 1, 500)

# Calcule les valeurs de la fonction de densité de probabilité
pdf_values = dist.pdf(x) # type: ignore

# Affiche la FDP
plt.figure(figsize=(10, 6))
plt.plot(x, pdf_values, label=f'Beta(alpha={alpha}, beta={beta_param})')
plt.title('Fonction de Densité de Probabilité (FDP) Bêta')
plt.xlabel('Valeur (x)')
plt.ylabel('Densité de Probabilité f(x)')
plt.grid(True)
plt.legend()
plt.show()

# Exemple: Probabilité que x soit près de 0 (par exemple, entre 0 et 0.1)
prob_near_zero = dist.cdf(0.1) - dist.cdf(0)
print(f"Probabilité d'être entre 0 et 0.1: {prob_near_zero:.4f}")

# Exemple: Probabilité que x soit près de 1 (par exemple, entre 0.9 et 1)
prob_near_one = dist.cdf(1) - dist.cdf(0.9)
print(f"Probabilité d'être entre 0.9 et 1: {prob_near_one:.4f}")