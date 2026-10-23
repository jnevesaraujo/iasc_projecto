###################################################################################
## Classe Ambiente10Accoes e script de teste do SARSA
###################################################################################
# Teste da biblioteca com o problema das 10 ações. O ambiente tem um único estado
# (s = 0) e cada ação a tem um valor real q*(a): a recompensa de cada tentativa é
# uma amostra de uma distribuição normal com média q*(a) e variância 1. O agente
# tem de estimar esses valores a partir das recompensas que vai observando.
# O script treina o SARSA durante 3000 episódios e verifica se a melhor ação
# (a ação 2, com q* = 2.0) foi aprendida.
#
###################################################################################
# q_estrela
# Valores médios reais q*(a) das 10 ações (índices 0 a 9). O máximo está na ação 2.
#
# iniciar()
# Devolve o estado inicial, sempre s = 0.
#
# executar(a)
# Gera a recompensa r = gauss(q_estrela[a], 1) e devolve (s_seguinte, r, final).
# O estado seguinte é 0 e final é True, porque cada episódio tem um único passo
# (logo J = r na aprendizagem).
#
# Script de teste
# - Cria a biblioteca: ε = 0.1, α = 0.1 e γ = 0.9.
# - Treina com sarsa.treinar(ambiente, 3000).
# - Imprime, para cada ação, o valor real q*(a) ao lado da estimativa Q(0, a).
# - Confirma com um assert que a ação com maior Q(0, a) é a ação 2.
###################################################################################
# Fontes:
#   MORGADO, Luis; Aprendizagem por reforço - Parte 1.pdf, ISEL 2026
#   Claude: Criação de Teste e Debugging
###################################################################################

from random import gauss

from memoria_aprendizagem import MemoriaAprendizagem
from aprendizagem import Aprendizagem
from seleccao_accao_sofrega import SeleccaoAccaoSofrega
from algoritmo_sarsa import AlgoritmoSarsa


class Ambiente10Accoes:
    q_estrela = [0.2, -0.8, 2.0, 0.4, 1.0, -1.5, -0.3, -0.9, 0.8, -0.2]

    def iniciar(self):
        return 0

    def executar(self, a):
        return 0, gauss(self.q_estrela[a], 1), True  # s', r, final


ambiente = Ambiente10Accoes()
accoes = list(range(10))

memoria = MemoriaAprendizagem()
selecao = SeleccaoAccaoSofrega(memoria, accoes, epsilon=0.1)
aprendizagem = Aprendizagem(memoria, alpha=0.1, gamma=0.9)
sarsa = AlgoritmoSarsa(selecao, memoria, aprendizagem)

sarsa.treinar(ambiente, 3000)

print("acção   q*     Q")
for a in accoes:
    print(f"{a:4d} {ambiente.q_estrela[a]:6.2f} {sarsa.Q(0, a):6.2f}")

melhor = max(accoes, key=lambda a: sarsa.Q(0, a))
assert melhor == 2, f"aprendeu a acção {melhor}, esperado 2"
print("OK: a melhor acção foi aprendida")