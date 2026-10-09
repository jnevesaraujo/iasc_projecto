from RedeNeuronal import RedeNeuronal
from FuncaoActivacao import Tanh

####################################################################################
## Aplicação: Classificação de Portas Lógicas
####################################################################################
# Demonstra a capacidade da RNA de aprender funções lógicas binárias, incluindo
# funções não linearmente separáveis (XOR, XNOR), que um perceptrão simples
# (uma só camada) não consegue aprender.
#
# Codificação bipolar: True = 1, False = -1 — adequada para Tanh cuja saída
# está em ]-1, 1[. A decisão final aplica um limiar em 0:
#   y >= 0 → True (1), y < 0 → False (-1)
#
# Arquitectura: [2, 4, 1]
#   - 2 entradas (A, B)
#   - 4 neurónios na camada escondida (igual ao exemplo XOR dos slides)
#   - 1 saída (resultado da porta)
#
# Parâmetros de treino:
#   alpha = 0.1  (taxa de aprendizagem)
#   beta  = 0.8  (momento — ajuda a atravessar regiões planas e mínimos locais)
#   nepocas    = 10000
#   epsilon_max = 0.05
####################################################################################

# Dados de treino — codificação bipolar (True=1, False=-1)
#   Cada linha: [A, B]
X = [[-1, -1],
     [-1,  1],
     [ 1, -1],
     [ 1,  1]]

# Tabelas de verdade para cada porta
#          A∧B   A∨B   A⊕B   A⊙B
portas = {
    "AND":  [[-1], [-1], [-1], [ 1]],
    "OR":   [[-1], [ 1], [ 1], [ 1]],
    "XOR":  [[-1], [ 1], [ 1], [-1]],
    "XNOR": [[ 1], [-1], [-1], [ 1]],
}

# Parâmetros
forma       = [2, 4, 1]
phi         = Tanh()
alpha       = 0.1
beta        = 0.8
nepocas     = 10000
epsilon_max = 0.05

print("=" * 52)
print(f"  RNA — Classificação de Portas Lógicas")
print(f"  Arquitectura : {forma}")
print(f"  Activação    : Tanh")
print(f"  alpha={alpha}  beta={beta}  epsilon_max={epsilon_max}")
print("=" * 52)

for nome, Y in portas.items():
    rede = RedeNeuronal(forma, phi)
    epsilon = rede.treinar(X, Y, nepocas, epsilon_max, alpha, beta)

    print(f"\n── {nome} ──")
    print(f"  Erro final : {epsilon:.6f}")
    print(f"  {'A':>3} {'B':>3}  {'Alvo':>6}  {'Saída':>8}  {'Decisão':>8}")
    print(f"  {'-'*3} {'-'*3}  {'-'*6}  {'-'*8}  {'-'*8}")

    for x, y in zip(X, Y):
        saida = rede.propagar(x)[0]
        decisao =  1 if saida >= 0 else -1
        correcto = "✓" if decisao == y[0] else "✗"
        print(f"  {x[0]:>3} {x[1]:>3}  {y[0]:>6}  {saida:>8.4f}  {decisao:>4}  {correcto}")
    print("\n")
    
print("\n" + "=" * 52)