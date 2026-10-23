class MemoriaAprendizagem:
    def __init__(self, valor_omissao = 0.0):
        """
        Iniciar memória de aprendizagem
        
        1. iniciar(𝑣𝑎𝑙𝑜𝑟_𝑜𝑚𝑖𝑠𝑠𝑎𝑜 = 0.0):
            2. 𝐭𝐡𝐢𝐬.𝑣𝑎𝑙𝑜𝑟_𝑜𝑚𝑖𝑠𝑠𝑎𝑜 = valor_omissao
            3. 𝑄 = {}                                   Memória de aprendizagem Q : (s, a) → ℝ
        """
        self.valor_omissao = valor_omissao
        self.Q = {}

    def Q(self, s, a):
        """
        Obter Q(s, a)
        
        1. q(s, a):
            2. returnQ.get((s, a), valor_omissao)
        """
        return self.Q.get((s, a), self.valor_omissao)

    def actualizar(self, s, a, q):
        """
        Actualizar memória de aprendizagem
        
        1. actualizar(s, a, q):
            2. Q[(s, a)] = q
        """
        self.Q[(s, a)] = q
        