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
#   de        — dimensão de entrada: nº de ligações por neuronio (= ds da camada anterior).
#   ds        — dimensão de saída: nº de neuronios da camada (largura da camada).
#   phi       — instância de FuncaoActivacao, partilhada por todos os neuronios.
#   neuronios — lista de ds objectos Neuronio, cada um inicializado com (de, phi).
#   y         — vector de saída actual da camada, inicializado a zeros (dimensão ds).
#               Guardado em memória após cada propagar para ser acessível durante
#               a retropropagação — a camada seguinte precisa de y^(n-1) para
#               calcular as correcções dos seus pesos (regra delta).
#
# y()
#   Devolve o vector de saída actual da camada — lista dos valores y de cada neuronio.
#
# propagar(x)
#   Invoca propagar(x) em cada neuronio — cada um calcula h = w·x + b e y = phi(h).
#   Guarda o resultado em self.y para ser consultado durante a retropropagação.
#   Retorna o vector y de dimensão ds, que se torna x^(n+1) da camada seguinte.
#
# adaptar(delta, y_anterior, alpha)
#   Delega a actualização de pesos e pendor a cada neuronio individualmente,
#   passando o componente de erro delta_j que lhe corresponde.
#   delta     — vector de erros da camada actual (um valor por neuronio).
#   y_anterior — vector de saída da camada anterior (entradas desta camada).
#   alpha     — taxa de aprendizagem.
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 1.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 2.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 3.pdf, ISEL 2026
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

    def y(self):
        """
        𝒚:
            2. return [𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜.𝑦 for 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜 in 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜𝑠]
        """
        return [neuronio.y for neuronio in self.neuronios]

    def propagar(self, x):
        """
        𝒙 - vector de entrada
        𝒚 - vector de saida 
        1. propagar 𝒙 :
            2. 𝒚 = [𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜.propagar 𝒙 for 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜 in 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜𝑠]
            3. return 𝒚
        """
        y = [neuronio.propagar(x) for neuronio in self.neuronios]
        self.y = y
        return y
    
    def adaptar(self, delta, y_anterior, alpha, beta):
        """
        𝜹^𝑛    Vector de erro de saída da camada 𝑛
        𝛿𝑗^𝑛   Componente 𝑗 do vector de erro de saída da camada 𝑛
        𝒚^𝑛−1  Vector de saída da camada anterior 𝑛 − 1
        𝛼      Taxa de aprendizagem
        𝛽      Factor de momento
        
        1. adaptar(𝜹^𝑛, 𝒚^𝑛−1, 𝛼, 𝛽):
            2. for 𝑗 = 1 to 𝑑𝑠:
            3. 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜𝑠[𝑗]. 𝑎𝑑𝑎𝑝𝑡𝑎𝑟(𝛿𝑗^𝑛, 𝒚^𝑛−1, 𝛼, 𝛽)
        """
        for neuronio, delta_j in zip(self.neuronios, delta):
            neuronio.adaptar(delta_j, y_anterior, alpha, beta)