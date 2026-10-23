###################################################################################
## Classe AlgoritmoSarsa
###################################################################################
# Coordena o ciclo de treino do SARSA (State-Action-Reward-State-Action), um método
# "On-Policy": a mesma política escolhe a ação a executar no ambiente e a ação
# seguinte a' usada no cálculo do retorno estimado. Assim, o que o agente aprende
# reflete o seu comportamento real, incluindo as ações de exploração.
#
###################################################################################
# __init__
# - seleccao_accao: escolhe a ação em cada estado a partir dos valores Q atuais
#   (ex.: estratégia ε-greedy).
# - memoria_aprendizagem: guarda e consulta os valores Q(s, a).
# - aprendizagem: aplica a regra de atualização por diferença temporal.
#
# Q(s, a)
# Devolve o valor estimado do par estado-ação, pedindo-o à memória de aprendizagem.
#
# treinar(ambiente, n_episodios)
# Por cada episódio:
# 1. Obtém o estado inicial s e seleciona a primeira ação a.
# 2. Repete, por cada passo, até o ambiente indicar um estado terminal (final):
#    - executa a ação a e observa s_seguinte, r e final;
#    - seleciona a_seguinte para s_seguinte, com a mesma política;
#    - chama aprender(s, a, r, s_seguinte, a_seguinte, final) para atualizar Q(s, a);
#    - avança: s = s_seguinte, a = a_seguinte.
###################################################################################
# Fontes:
#   MORGADO, Luis; Aprendizagem por reforço - Parte 1.pdf, ISEL 2026
#   Claude: Debugging
###################################################################################

class AlgoritmoSarsa:
    """
    Controlo de aprendizagem com política de selecção de acção única (“On-Policy”)
    
    1. Iniciar Q(s, a)
    2. Repetir (por cada episódio)
        3. Iniciar s
        4. Seleccionar acção a para o estado s
        5. Repetir (por cada passo)
        6. Executar acção a, observar r e s’
        7. Seleccionar acção a’ para o estado s’
        8. aprender(s, a, r, s’, a’)
        9. Actualizar s  s’, a  a’
        10. Até s ser um estado terminal
    """
    
    def __init__(self, seleccao_accao, memoria_aprendizagem, aprendizagem):
        self.sel_accao = seleccao_accao
        self.mem_aprendizagem = memoria_aprendizagem
        self.aprendizagem = aprendizagem

    def Q(self, s, a):
        return self.mem_aprendizagem.Q(s, a)
        
    def treinar(self, ambiente, n_episodios):
        for _ in range(n_episodios):                          
            s = ambiente.iniciar()                            
            a = self.sel_accao.seleccionar_accao(s)                    
            final = False
            while not final:                                  
                s_seguinte, r, final = ambiente.executar(a)       
                a_seguinte = self.sel_accao.seleccionar_accao(s_seguinte)     
                self.aprendizagem.aprender(s, a, r, s_seguinte, a_seguinte, final)   
                s, a = s_seguinte, a_seguinte
  

