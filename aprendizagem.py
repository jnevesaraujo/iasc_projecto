class Aprendizagem:
    def __init__(self):
        pass
    
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
        self.Q(s, a) = self.Q(s, a) + self.alpha * (J - self.Q(s, a))