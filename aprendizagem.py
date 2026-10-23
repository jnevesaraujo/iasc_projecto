###################################################################################
## Classe Aprendizagem
###################################################################################
# Implementa o passo de aprendizagem do SARSA (on-policy), por diferença temporal.
# A cada transição (s, a, r, s', a') a estimativa Q(s, a) é ajustada em direção ao
# retorno estimado J, de modo a convergir para o valor real de cada ação.
#
###################################################################################
# __init__
# - memoria_aprendizagem: onde se guardam e consultam os valores Q(s, a).
# - alpha [0,1] (factor de aprendizagem): peso das observações novas.
#   Perto de 1 conta o mais recente; perto de 0 mantém a estimativa anterior.
# - gamma [0,1] (factor de desconto): peso das recompensas futuras.
#   Perto de 0 o agente é imediatista; perto de 1 valoriza o longo prazo.
#
# aprender(s, a, r, s_seguinte, a_seguinte, final)  
# 1. Retorno estimado J:
#    - final: não há ação seguinte, logo J = r.
#    - caso contrário:   J = r + gamma * Q(s', a'), a recompensa imediata mais o
#      valor descontado do par seguinte (retorno estimado).
# 2. Atualização por diferença temporal :
#    Q(s, a) = Q(s, a) + alpha * (J - Q(s, a))
#    em que (J - Q) é o erro de estimativa, a diferença entre o valor alvo e a
#    estimativa anterior.
###################################################################################
# Fontes:
#   MORGADO, Luis; Aprendizagem por reforço - Parte 1.pdf, ISEL 2026
#   Claude: Debugging 
###################################################################################

class Aprendizagem:
    def __init__(self, memoria_aprendizagem, alpha, gamma):
        self.memoria_aprendizagem = memoria_aprendizagem
        self.alpha = alpha
        self.gamma = gamma
    
    def aprender(self, s, a, r, s_seguinte, a_seguinte, final):
        """
        1. aprender(s, a, r, s’, a’, final):
            2. if final:
            3. J = r Retorno estimado do passo final
            4. else:
            5. J = r + γQ(s’, a’)
            6. Q(s, a) = Q(s, a) + α(J – Q(s, a))
        """
        
        if final:
            J = r
        else:
            J = r + self.gamma * self.Q(s_seguinte, a_seguinte)
        
        q = self.memoria_aprendizagem.Q(s, a)
        self.memoria_aprendizagem.actualizar(s, a, q + self.alpha * (J - q))