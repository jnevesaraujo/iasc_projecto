from Neuronio import Neuronio

###################################################################################
## Classe CamadaDensa
###################################################################################
# A camada densa representa uma camada escondida ou de saída da RNA multicamada.
# Cada neuronio está ligado a todas as saídas da camada anterior (fully connected),
# recebendo o vector de entradas x (dimensão de) e produzindo y (dimensão ds).
#
# Ao contrário da CamadaEntrada, aplica transformações não-lineares através das
# funções de activação dos seus neuronios; é aqui que a rede projecta os dados
# em espaços de maior dimensionalidade, criando representações progressivamente
# mais abstractas que permitem separar classes não separáveis linearmente.
#
# O processamento segue o modelo matricial dos slides:
#   h^n = x^n · W^n + b^n   (soma de activação de todos os neuronios da camada)
#   y^n = phi^n(h^n)         (saída da camada, entrada da camada seguinte)
#
# __init__(de, ds, phi)
#   de       — dimensão de entrada: nº de ligações por neuronio (= ds da camada anterior).
#   ds       — dimensão de saída: nº de neuronios da camada (largura da camada).
#   phi      — função de activação partilhada por todos os neuronios.
#   neuronios — lista de ds objectos Neuronio, cada um inicializado com (de, phi).
#
# y()
#   Devolve o vector de saída actual da camada — lista dos valores y de cada neuronio.
#
# propagar(x)
#   Invoca propagar(x) em cada neuronio — cada um calcula h = w·x + b e y = phi(h).
#   Retorna o vector y de dimensão ds, que se torna x^(n+1) da camada seguinte.
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 1.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 2.pdf, ISEL 2026
###################################################################################

class CamadaDensa:
    def __init__(self, de, ds, phi):
        """
        iniciar(𝑑𝑒, 𝑑𝑠,𝜑):
            2. 𝑑𝑒 ∈ 𝑍+                              Dimensão de entrada (número ligações de entrada)
            3. 𝑑𝑠 ∈ 𝑍+                              Dimensão de saída (número de saídas da camada)
            4. 𝜑: 𝑅 → 𝑅                             Função de activação
            5. 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜𝑠 = [Neuronio 𝑑𝑒,𝜑 for 𝑑𝑠]    Neurónios da camada
        """
        self.de = de
        self.ds = ds
        self.phi = phi
        self.neuronios = [Neuronio(de, phi) for _ in range(ds)]

   # def y(self):
        """
        𝒚:
            2. return [𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜.𝑦 for 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜 in 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜𝑠]
        """
        #return [neuronio.y for neuronio in self.neuronios]

    def propagar(self, x):
        """
        𝒙 - vector de entrada
        𝒚 - vector de saida 
        1. propagar 𝒙 :
            2. 𝒚 = [𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜.propagar 𝒙 for 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜 in 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜𝑠]
            3. return 𝒚
        """
        y = [neuronio.propagar(x) for neuronio in self.neuronios]
        return y
    
    def adaptar(self, delta, y_anterior, alpha):
        """
        1. adaptar(𝛿, 𝒚^𝑛−1, 𝛼):
            2. Δ𝒘= −𝛼𝑦′𝛿 ∙ 𝒚^𝑛−1 Variação dos pesos
            3. 𝒘 = 𝒘 + Δ𝒘 Actualização dos pesos
            4. Δ𝑏= −𝛼𝑦′𝛿 Variação do pendor
            5. 𝑏 = 𝑏 + Δ𝑏 Actualização do pendor
        """
        for neuronio, delta_j in zip(self.neuronios, delta):
            neuronio.adaptar(delta_j, y_anterior, alpha)