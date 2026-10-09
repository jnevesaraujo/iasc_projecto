import math
from RedeNeuronal import RedeNeuronal
from FuncaoActivacao import Tanh
from pathlib import Path

####################################################################################
## Aplicação: Classificação de Iris (Setosa vs Versicolor)
####################################################################################
# Demonstra a aplicação da RNA a um problema real de classificação binária
# com dados contínuos. O dataset Iris contém 4 medidas de pétalas e sépalas
# de duas espécies de flores; a rede aprende a distingui-las.
#
# Pré-processamento:
#   - Normalização min-max para [-1, 1]: x' = 2*(x - min)/(max - min) - 1
#     Necessária para que nenhuma feature domine pela sua escala, e para
#     compatibilidade com a saída da Tanh ∈ ]-1, 1[.
#   - Codificação bipolar: Setosa = 1, Versicolor = -1
#
# Divisão treino/teste:
#   - 80% treino (40 Setosa + 40 Versicolor = 80 amostras)
#   - 20% teste  (10 Setosa + 10 Versicolor = 20 amostras)
#   - Intercalado para garantir proporção igual de classes em ambos os conjuntos
#
# Arquitectura: [4, 8, 1]
#   - 4 entradas (comprimento/largura da sépala e pétala)
#   - 8 neurónios na camada escondida
#   - 1 saída: >= 0 → Setosa, < 0 → Versicolor
#
# Parâmetros:
#   alpha = 0.1, beta = 0.8, nepocas = 5000, epsilon_max = 0.05
####################################################################################


def carregar_iris(caminho):
    """
    Carrega o CSV e devolve (X, Y) com codificação bipolar.
    Cada linha do CSV tem 5 colunas:
        0: comprimento sépala
        1: largura sépala
        2: comprimento pétala
        3: largura pétala
        4: espécie (Setosa ou Versicolor)
    """
    X, Y = [], []
    with open(caminho) as f:
        for linha in f:
            partes = linha.strip().split(",")
            x = [float(v) for v in partes[:4]]
            y = [1] if partes[4] == "Iris-setosa" else [-1]
            X.append(x)
            Y.append(y)
    return X, Y


def normalizar(X):
    """
    Normalização min-max para [-1, 1] por feature, calculada no conjunto de treino.
    Retorna a função de normalização e o conjunto normalizado.
    """
    n_features = len(X[0])
    mins = [min(x[i] for x in X) for i in range(n_features)]
    maxs = [max(x[i] for x in X) for i in range(n_features)]
    def norm(x):
        return [2 * (x[i] - mins[i]) / (maxs[i] - mins[i]) - 1
                for i in range(n_features)]
    return norm, [norm(x) for x in X]


def dividir_treino_teste(X, Y, proporcao_treino=0.8):
    """
    Divide intercalando para manter proporção igual de classes.
    Com 50 Setosa + 50 Versicolor e 80% treino:
      treino = 40 Setosa + 40 Versicolor = 80 amostras
      teste  = 10 Setosa + 10 Versicolor = 20 amostras
    """
    X_set, Y_set = X[:50],  Y[:50]   # Setosa
    X_ver, Y_ver = X[50:], Y[50:]    # Versicolor
    n_treino = int(50 * proporcao_treino)

    X_treino = X_set[:n_treino] + X_ver[:n_treino]
    Y_treino = Y_set[:n_treino] + Y_ver[:n_treino]
    X_teste  = X_set[n_treino:] + X_ver[n_treino:]
    Y_teste  = Y_set[n_treino:] + Y_ver[n_treino:]
    return X_treino, Y_treino, X_teste, Y_teste


def avaliar(rede, X, Y):
    """
    Calcula accuracy: percentagem de amostras correctamente classificadas.
    rede    instância de RedeNeuronal
    𝑿       Conjunto de entradas 𝒙
    𝒀       Conjunto de saídas 𝒚
    """
    correcto = 0
    for x, y in zip(X, Y):
        saida = rede.propagar(x)[0]
        decisao = 1 if saida >= 0 else -1
        if decisao == y[0]:
            correcto += 1
    return correcto / len(Y) * 100


# --- Carregar e preparar dados ---
dataset = Path(__file__).parent / "dataset" / "iris.csv"
X, Y = carregar_iris(dataset)

X_treino, Y_treino, X_teste, Y_teste = dividir_treino_teste(X, Y)

# Normalizar com base nos dados de treino
norm_fn, X_treino_norm = normalizar(X_treino)
X_teste_norm = [norm_fn(x) for x in X_teste]

# --- Treinar ---
forma       = [4, 8, 1]
phi         = Tanh()
alpha       = 0.1
beta        = 0.8
nepocas     = 5000
epsilon_max = 0.05

rede = RedeNeuronal(forma, phi)

print("=" * 52)
print(f"  RNA — Classificação Iris (Setosa vs Versicolor)")
print(f"  Arquitectura : {forma}")
print(f"  Treino: {len(X_treino)} amostras  |  Teste: {len(X_teste)} amostras")
print(f"  alpha={alpha}  beta={beta}  epsilon_max={epsilon_max}")
print("=" * 52)

epsilon = rede.treinar(X_treino_norm, Y_treino, nepocas, epsilon_max, alpha, beta)
print(f"  Erro final treino : {epsilon:.6f}")

# --- Avaliar ---
acc_treino = avaliar(rede, X_treino_norm, Y_treino)
acc_teste  = avaliar(rede, X_teste_norm,  Y_teste)

print(f"  Accuracy treino   : {acc_treino:.1f}%")
print(f"  Accuracy teste    : {acc_teste:.1f}%")

# --- Resultados detalhados do conjunto de teste ---
print(f"\n  {'#':>3}  {'Classe':>12}  {'Saída':>8}  {'Decisão':>12}  {'':>2}")
print(f"  {'-'*3}  {'-'*12}  {'-'*8}  {'-'*12}  {'-'*2}")
for i, (x, y) in enumerate(zip(X_teste_norm, Y_teste)):
    saida    = rede.propagar(x)[0]
    decisao  = 1 if saida >= 0 else -1
    classe   = "Setosa" if decisao == 1 else "Versicolor"
    alvo     = "Setosa" if y[0] == 1   else "Versicolor"
    correcto = "✓" if decisao == y[0] else "✗"
    print(f"  {i+1:>3}  {alvo:>12}  {saida:>8.4f}  {classe:>12}  {correcto}")
print("\n")

print("=" * 52)