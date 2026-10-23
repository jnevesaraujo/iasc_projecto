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
        self.seleccao_accao = seleccao_accao
        self.memoria_aprendizagem = memoria_aprendizagem
        self.aprendizagem = aprendizagem

    def Q(self, s, a):
        return self.memoria_aprendizagem.Q(s, a)

    def seleccionar_accao(self, s):
        return self.seleccao_accao.seleccionar_accao(s)

    def aprender(self, s, a, r, s_seguinte, a_seguinte, final):
        self.aprendizagem.aprender(s, a, r, s_seguinte, a_seguinte, final)
  

