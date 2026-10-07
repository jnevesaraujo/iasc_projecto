import math

import math

####################################################################################
## Interface FuncaoActivacao — φ(h) e dφ(h)/dh
####################################################################################
# A função de activação φ introduz não-linearidade em cada neuronio, permitindo
# à rede aprender fronteiras de decisão não lineares.
#
# A retropropagação exige não só φ(h) mas também a sua derivada dφ(h)/dh,
# usada na regra delta para propagar o erro e actualizar os pesos:
#   Δw = -α · y' · δ · y^(n-1)   onde y' = dφ(h)/dh
#
# A classe FuncaoActivacao define a interface comum: __call__ para φ(h) e
# derivada para dφ(h)/dh. Cada subclasse implementa o par analítico concreto,
# o que é preferível à aproximação numérica por diferenças finitas — é exacto,
# mais eficiente e evita erros de arredondamento acumulados ao longo do treino.
#
# Tanh    — φ(h) = tanh(h),              saída ∈ ]−1, 1[. Função de referência
#            dos slides; adequada para codificação bipolar (alvos −1 e 1).
# Sigmoid — φ(h) = 1 / (1 + e^(−h)),    saída ∈ ]0, 1[. Interpretável como
#            probabilidade; útil quando os alvos são 0 e 1.
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 3.pdf, ISEL 2026
####################################################################################

class FuncaoActivacao:
    def __call__(self, h):
        """φ(h) — função de activação aplicada à soma de activação h."""
        raise NotImplementedError

    def derivada(self, h):
        """dφ(h)/dh — derivada analítica, necessária para retropropagar."""
        raise NotImplementedError


class Tanh(FuncaoActivacao):
    """
    φ(h)     = tanh(h)          saída ∈ ]−1, 1[
    dφ(h)/dh = 1 − tanh²(h)    derivada analítica
    """
    def __call__(self, h):
        return math.tanh(h)

    def derivada(self, h):
        return 1 - math.tanh(h) ** 2


class Sigmoid(FuncaoActivacao):
    """
    φ(h)     = 1 / (1 + e^(−h))     saída ∈ ]0, 1[
    dφ(h)/dh = φ(h) · (1 − φ(h))    derivada expressa em função da própria saída
    """
    def __call__(self, h):
        return 1 / (1 + math.exp(-h))

    def derivada(self, h):
        s = self(h)
        return s * (1 - s)

""" 
class ReLU(FuncaoActivacao):
    def __call__(self, h):
        return max(0.0, h)
    
    def derivada(self, h):
        return 1.0 if h > 0 else 0.0 """