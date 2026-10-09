from FuncaoActivacao import Tanh
from RedeNeuronal import RedeNeuronal

###################################################################################
## Teste — Reconhecimento de Barras Verticais
## Parte 2 — Retropropagação
###################################################################################
# Valida a implementação do algoritmo de retropropagação (slides 33-35, Parte 2).
# A rede aprende a distinguir imagens 3x3 com barras verticais (saída=1)
# de imagens sem barras verticais (saída=0).
#
# Cada imagem é representada por 9 valores (um por pixel), lidos linha a linha.
# A rede [9, 3, 1] tem 9 entradas, 3 neuronios escondidos e 1 neuronio de saída.
# Função de activação: tanh — admite valores negativos, adequada para codificação
# bipolar, e é diferenciável (necessário para retropropagação).
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 3.pdf, ISEL 2026
###################################################################################


# Dados de treino: 7 imagens 3x3 codificadas como vector de 9 valores
X = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],  # vazia       → 0
    [1, 0, 0, 1, 0, 0, 1, 0, 0],  # coluna 1    → 1
    [0, 1, 0, 0, 1, 0, 0, 1, 0],  # coluna 2    → 1
    [0, 0, 1, 0, 0, 1, 0, 0, 1],  # coluna 3    → 1
    [1, 1, 1, 0, 0, 0, 0, 0, 0],  # linha 1     → 0
    [0, 0, 0, 1, 1, 1, 0, 0, 0],  # linha 2     → 0
    [0, 0, 0, 0, 0, 0, 1, 1, 1],  # linha 3     → 0
]
Y = [[0], [1], [1], [1], [0], [0], [0]]

# Rede [9, 3, 1] com tanh
rede = RedeNeuronal([9, 3, 1], Tanh())

# Parâmetros de treino:
# alpha=0.2, épocas=1000, erro_max=0.05
rede.treinar(X, Y, nepocas=1000, epsilon_max=0.05, alpha=0.2, beta=0.8)

# Verificar resultados
print("\nEntrada | Alvo | Obtido")
print("-" * 35)
Y_obtido = rede.prever(X)
for x, alvo, obtido in zip(X, Y, Y_obtido):
    print(f"  {x}  |  {alvo}  |  {[round(v, 4) for v in obtido]}")