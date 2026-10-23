from random import random, shuffle

class SeleccaoAccaoSofrega:
    def __init__(self):
        pass
    
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
        shuffle(self.accoes)
        return max(self.Q(s, a) for a in self.accoes)    
    
    def accao_exploracao(self):
        """
        1. accao_exploracao():
            2. return choice(accoes)
        """
        return random.choice(self.accoes)