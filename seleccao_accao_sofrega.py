###################################################################################
## Classe SeleccaoAccaoSofrega
###################################################################################
# Implementa a estratégia de selecção de acção ε-greedy, que faz o compromisso entre
# exploração e aproveitamento:
# - com probabilidade 1 - ε, aproveita o conhecimento adquirido e escolhe a acção
#   sôfrega (greedy), a de maior valor estimado Q(s, a);
# - com probabilidade ε, explora, escolhendo uma acção ao acaso para ganhar
#   experiência e melhorar a aprendizagem.
#
###################################################################################
# __init__
# - memoria_aprendizagem: guarda e consulta os valores Q(s, a).
# - accoes: conjunto de acções que o agente pode executar.
# - epsilon [0,1]: probabilidade de exploração.
#
# seleccionar_accao(s)
# Gera um número aleatório com random(). Se for superior a ε, devolve a acção
# sôfrega (aproveitamento); caso contrário, devolve uma acção de exploração.
#
# accao_sofrega(s)
# Devolve a acção com maior Q(s, a), ou seja, argmax Q(s, a). Antes disso baralha
# as acções com o shuffle, para que o desempate entre estas com o mesmo valor seja aleatório.
#
# accao_exploracao()
# Devolve uma acção escolhida ao acaso com choice(accoes), todas com a mesma
# probabilidade.
###################################################################################
# Fontes:
#   MORGADO, Luis; Aprendizagem por reforço - Parte 1.pdf, ISEL 2026
#   Claude: Debugging
###################################################################################

from random import random, shuffle, choice

class SeleccaoAccaoSofrega:
    def __init__(self, memoria_aprendizagem, accoes, epsilon):
        self.memoria_aprendizagem = memoria_aprendizagem
        self.accoes = accoes
        self.epsilon = epsilon
    
    def seleccionar_accao(self, s):
        """
        1. seleccionar_accao(s):
            2. if random() > ε:
            3. a = accao_sofrega(s)     Aproveitar conhecimento
            4. else:
            5. a = accao_exploracao()   Explorar
            6. return a
        """
        if random() > self.epsilon:
            a = self.accao_sofrega(s)
        else:
            a = self.accao_exploracao()
        return a

    def accao_sofrega(self, s):
        """
        1. accao_sofrega(s):
            2. shuffle(accoes)          Escolha aleatória entre acções com igual valor
            3. return argmax 𝑄(𝑠, 𝑎)
            𝑎∈𝑎𝑐𝑐𝑜𝑒𝑠
            
        """
        accoes = list(self.accoes)
        shuffle(self.accoes)
        return max(accoes, key=lambda a: self.memoria_aprendizagem.Q(s, a))    
    
    def accao_exploracao(self):
        """
        1. accao_exploracao():
            2. return choice(accoes)
        """
        return choice(self.accoes)