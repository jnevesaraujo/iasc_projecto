###################################################################################
## Classe MemoriaAprendizagem
###################################################################################
# Implementa a estrutura de dados que armazena e gere a função de valor de acção
# Q : (s, a) → ℝ, ou seja, a estimativa da utilidade/qualidade esperada de
# realizar a acção a no estado s. Representa o conhecimento acumulado pelo agente.
# É um dicionário, por isso só ocupa memória para os pares estado-acção que já
# foram actualizados; os restantes assumem um valor por omissão.
#
###################################################################################
# __init__
# - valor_omissao: valor de Q para pares (s, a) ainda não visitados (por omissão, 0.0).
# - memoria: dicionário com chave (s, a) e valor Q(s, a).
#
# Q(s, a)
# Devolve a estimativa Q(s, a) com memoria.get((s, a), valor_omissao): se o par já
# existe devolve o valor guardado, senão devolve o valor por omissão.
#
# actualizar(s, a, q)
# Guarda a nova estimativa: memoria[(s, a)] = q.
###################################################################################
# Fontes:
#   MORGADO, Luis; Aprendizagem por reforço - Parte 1.pdf, ISEL 2026
#   Claude: Debugging
###################################################################################

class MemoriaAprendizagem:
    def __init__(self, valor_omissao = 0.0):
        """
        Iniciar memória de aprendizagem
        
        1. iniciar(𝑣𝑎𝑙𝑜𝑟_𝑜𝑚𝑖𝑠𝑠𝑎𝑜 = 0.0):
            2. 𝐭𝐡𝐢𝐬.𝑣𝑎𝑙𝑜𝑟_𝑜𝑚𝑖𝑠𝑠𝑎𝑜 = valor_omissao
            3. 𝑄 = {}                                   Memória de aprendizagem Q : (s, a) → ℝ
        """
        self.valor_omissao = valor_omissao
        self.memoria = {}

    def Q(self, s, a):
        """
        Obter Q(s, a)
        
        1. q(s, a):
            2. return Q.get((s, a), valor_omissao)
        """
        return self.memoria.get((s, a), self.valor_omissao)

    def actualizar(self, s, a, q):
        """
        Actualizar memória de aprendizagem
        
        1. actualizar(s, a, q):
            2. Q[(s, a)] = q
        """
        self.memoria[(s, a)] = q
        