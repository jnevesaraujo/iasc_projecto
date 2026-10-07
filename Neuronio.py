from random import random

###################################################################################
## Classe Neuronio
###################################################################################
# O neuronio artificial é a unidade básica de processamento da RNA, inspirado
# no neuronio biológico; as entradas x correspondem aos sinais das dendrites,
# os pesos w à permeabilidade sináptica, o pendor b ao limiar de activação,
# e a saída y ao sinal transmitido pelo axónio.
#
# Cada neuronio define uma fronteira de discriminação (hiperplano) no espaço
# das entradas: os pesos controlam o declive (orientação) e o pendor controla
# a deslocação dessa fronteira.
#
# A função de transferência é f(x) = phi(h), onde:
#   h = w·x + b  (soma de activação — produto escalar das entradas pelos pesos)
#   y = phi(h)   (saída — função de activação aplicada a h)
#
# A retropropagação exige a derivada da função de activação em cada neuronio:
#   y' = dphi(h)/dh  (derivada da saída em ordem à soma de activação)
# A derivada é calculada analiticamente através da interface FuncaoActivacao,
# que encapsula phi e dphi/dh numa só entidade.
#
# __init__(d, phi)
#   d   — número de ligações de entrada (dimensão do vector x).
#   phi — função de activação: instância de FuncaoActivacao, que expõe
#         phi(h) via __call__ e a derivada analítica via phi.derivada(h).
#   w   — vector de pesos, inicializado aleatoriamente em [-1, 1].
#   b   — pendor, inicializado aleatoriamente em [-1, 1].
#   h   — soma de activação, inicializado a 0.
#   y   — saída do neuronio, inicializado a 0.
#   dy  — derivada da saída (y' = dphi(h)/dh), inicializado a 0;
#         guardado após propagar para ser reutilizado em retropropagar.
#
# propagar(x)
#   Calcula h = w·x + b via produto escalar (zip entre w e x).
#   Aplica y = phi(h) — a não-linearidade que permite discriminar padrões.
#   Calcula y' = phi.derivada(h) — derivada analítica necessária para
#   retropropagar o erro e actualizar os pesos (regra da cadeia).
#   Retorna y para ser usado como entrada da camada seguinte (x^n = y^(n-1)).
#
# adaptar(delta, y_anterior, alpha)
#   Actualiza os pesos e o pendor do neuronio pelo gradiente descendente,
#   aplicando a regra delta local derivada da retropropagação:
#     Δw = -α · y' · δ · y^(n-1)   (correcção proporcional à entrada recebida)
#     Δb = -α · y' · δ              (correcção do pendor, independente da entrada)
#   delta     — componente de erro propagado até este neuronio pela camada seguinte.
#   y_anterior — vector de saída da camada anterior, que foi a entrada deste neuronio.
#   alpha     — taxa de aprendizagem: controla a dimensão do passo de actualização.
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 1.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 2.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 3.pdf, ISEL 2026
###################################################################################

class Neuronio:
    def __init__(self, d, phi):
        """
        1. iniciar(𝑑,𝜑):
            2. 𝑑 ∈ 𝑍                    Dimensão de entrada (número ligações de entrada)
            3. 𝜑: 𝑅 → 𝑅                 Função de activação
            4. 𝒘 = random(𝑑, −1, 1)     Vector de pesos das ligações de entrada
            5. 𝑏 = random(−1, 1)        Pendor interno do neurónio
            6. ℎ = 0                    Soma da activação das entradas
            7. 𝑦 = 0                    Saída do neurónio
            8. y' = 0                   Derivada da saída do neurónio
        """

        self.d = d
        self.phi = phi
        self.w =  [random() * 2 - 1 for _ in range(d)]      # random(d, -1, 1)
        self.b = random() * 2 - 1                           # random(-1, 1)
        self.h = 0
        self.y = 0
        self.dy = 0                                         # y' inicializado a 0


    def propagar(self, x):
        """
        1. propagar (𝒙) :               𝒙 - vector de entradas
            2. ℎ = 𝒘 ∙ 𝒙 + 𝑏
            3. 𝑦 = 𝜑 ℎ
            4. 𝑦′ = 𝑑𝜑(ℎ)/𝑑ℎ
            5. return 𝑦
        """
        self.h = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b         # w·x + b
        self.y = self.phi(self.h)                                           # φ(h)
        e = 1e-5
        self.dy = self.phi.derivada(self.h)
#        self.dy = (self.phi(self.h + e) - self.phi(self.h - e)) / (2 * e)   # 𝑑𝜑(ℎ)/𝑑ℎ ≅ 𝜑(ℎ + ε) - 𝜑(ℎ - ε) / (2ε)
        return self.y
    
    def adaptar(self, delta, y_anterior, alpha):
        """
        1. adaptar(𝛿, 𝒚^𝑛−1, 𝛼):
            2. Δ𝒘= −𝛼𝑦′𝛿 ∙ 𝒚^𝑛−1    Variação dos pesos
            3. 𝒘 = 𝒘 + Δ𝒘          Actualização dos pesos
            4. Δ𝑏= −𝛼𝑦′𝛿            Variação do pendor
            5. 𝑏 = 𝑏 + Δ𝑏           Actualização do pendor

        𝛿 Componente de propagação do erro de saída do neurónio
        𝒚^𝑛−1 Vector de saída da camada anterior 𝑛−1
        𝛼 Taxa de aprendizagem
        """
        dw = [-alpha * self.dy * delta * yi for yi in y_anterior]
        self.w = [wi + dwi for wi, dwi in zip(self.w, dw)]
        self.b = self.b + (-alpha * self.dy * delta)