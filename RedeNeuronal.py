from CamadaDensa import CamadaDensa
from CamadaEntrada import CamadaEntrada

###################################################################################
## Classe RedeNeuronal
###################################################################################
# A RedeNeuronal representa um Perceptrão Multicamada (MLP), gerindo a topologia
# completa da rede e a propagação sequencial de informação (activação directa).
#
# Organiza as camadas em cadeia: uma CamadaEntrada seguida de N-1 CamadasDensas,
# onde a saída de cada camada se torna a entrada da seguinte (x^n = y^(n-1)).
# Esta arquitectura permite à rede realizar transformações não-lineares sucessivas,
# projectando os dados em espaços de dimensões superiores onde classes não
# separáveis linearmente se tornam separáveis (Teorema da Aproximação Universal).
#
# Os parâmetros da rede são: theta = {W^2, b^2, ..., W^N, b^N}
# (não inclui a camada de entrada, que não tem pesos associados).
#
# __init__(forma, phi, pesos=None)
#   forma — lista de inteiros: cada elemento é o nº de neuronios da camada.
#            ex: [2, 2, 1] → entrada(2), escondida(2), saída(1).
#   phi   — instância de FuncaoActivacao, partilhada por todas as camadas densas.
#   pesos — opcional: lista de (W, b) por camada densa para substituir os
#            pesos aleatórios. Usado para replicar os exemplos dos slides.
#
# propagar(x)
#   Executa o forward pass: y é inicializado com x e transformado sequencialmente
#   por cada camada. A saída final y^N é a resposta da rede para a entrada x.
#   Cada camada guarda self.y após propagar — necessário para retropropagar.
#
# prever(X)
#   Aplica propagar(x) a cada amostra x do conjunto X (inferência em lote).
#   Retorna Y — lista de respostas correspondentes.
#
# delta_saida(yN, y)
#   Calcula o vector de erro de saída da rede: delta^N = y^N - y.
#   É o ponto de partida da retropropagação — mede o desvio entre o que
#   a rede produziu e o que deveria ter produzido.
#
# retropropagar(deltaN, alpha)
#   Propaga o erro da camada de saída para a camada de entrada (backward pass),
#   actualizando os pesos e pendores de cada camada densa pela regra delta.
#   Para cada camada n (de N até 2):
#     - consulta y^(n-1) guardado na camada anterior (necessário para adaptar)
#     - calcula delta^(n-1) pela regra da cadeia: soma ponderada dos erros
#       da camada n, ponderada pelos pesos e derivadas dos seus neuronios
#     - chama adaptar na camada n para actualizar os seus pesos
#     - passa delta^(n-1) para a iteração seguinte
#   A camada de entrada (n=1) não tem pesos — o ciclo termina em n=2.
#
# adaptar(x, y, alpha)
#   Executa um passo completo de aprendizagem para um par (x, y):
#     1. propaga x para obter y^N
#     2. calcula delta^N = delta_saida(y^N, y)
#     3. retropropaga o erro e actualiza os pesos
#     4. devolve o erro quadrático médio epsilon = (1/K) * sum(delta_k^N²)
#   epsilon mede a qualidade da resposta da rede para esta amostra.
#
# treinar(X, Y, nepocas, epsilon_max, alpha)
#   Executa o ciclo de treino supervisionado por épocas:
#   cada época percorre todos os pares (x, y) e chama adaptar.
#   O erro da época é o máximo dos erros individuais (critério de paragem
#   conservador — garante que nenhuma amostra fica mal classificada).
#   Para antecipadamente se epsilon <= epsilon_max (convergência).
#
# definir_pesos(pesos)
#   Método auxiliar de validação e testes:
#   Permite substituir os pesos e pendores gerados aleatoriamente por matrizes
#   e vectores de pesos (W, b), fornecidos manualmente para cada camada densa.
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 1.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 2.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 3.pdf, ISEL 2026
###################################################################################

class RedeNeuronal:
    def __init__(self, forma, phi, pesos=None):
        """
        𝑓𝑜𝑟𝑚𝑎 Forma da rede (lista de número de neurónios por camada)
        𝜑 Função de activação dos neurónios das camadas
        𝑁 Número de camadas
        𝑑𝑒^𝑛 Dimensão de entrada da camada 𝑛
        𝑑𝑠^𝑛 Dimensão de saída da camada 𝑛

        1. iniciar(𝑓𝑜𝑟𝑚𝑎, 𝜑):
            2. 𝑐𝑎𝑚𝑎𝑑𝑎𝑠 = [ ]
            3. 𝑁 = length(𝑓𝑜𝑟𝑚𝑎)
            4. 𝑑𝑠^1 = 𝑓𝑜𝑟𝑚𝑎[1]                      Dimensão de entrada
            5. 𝑐𝑎𝑚𝑎𝑑𝑎^1 = CamadaEntrada(𝑑𝑠^1)       Camada de entrada
            6. 𝑐𝑎𝑚𝑎𝑑𝑎𝑠.append(𝑐𝑎𝑚𝑎𝑑𝑎^1)
            7. for 𝑛 = 2 to 𝑁:
                8. 𝑑𝑒^𝑛 = 𝑓𝑜𝑟𝑚𝑎[𝑛 − 1]
                9. 𝑑𝑠^𝑛 = 𝑓𝑜𝑟𝑚𝑎 [𝑛]
                10. 𝑐𝑎𝑚𝑎𝑑𝑎^𝑛 = CamadaDensa(𝑑𝑒^𝑛, 𝑑𝑠^𝑛,𝜑)
                11. 𝑐𝑎𝑚𝑎𝑑𝑎𝑠.append(𝑐𝑎𝑚𝑎𝑑𝑎^𝑛)
        """
        self.camadas = []
        self.N = len(forma)
        ds = forma[0]
        camada1 = CamadaEntrada(ds)
        self.camadas.append(camada1)
        for n in range(1, self.N):
            de = forma[n-1]
            ds = forma[n]
            camada = CamadaDensa(de, ds, phi)
            self.camadas.append(camada)
        if pesos:
            self.definir_pesos(pesos)
            
    def prever(self, X):
        """
        𝑿 Conjunto de vectores de entrada
        𝒀 Conjunto de vectores de saída
        𝒙 Vector de entrada

        1. prever 𝑿 :
            2. 𝒀 = [propagar 𝒙 for 𝒙 ∈ 𝑿]
            3. return 𝒀
        """

        self.Y = [self.propagar(x) for x in X]
        return self.Y

    def propagar(self, x):
        """
        1. propagar 𝒙 :
            2. 𝒚 = 𝒙
            3. for 𝑐𝑎𝑚𝑎𝑑𝑎 in 𝑐𝑎𝑚𝑎𝑑𝑎𝑠:
                4. 𝒚 = 𝑐𝑎𝑚𝑎𝑑𝑎. propagar(𝒚)
            5. return 𝒚
        """
        y = x
        for camada in self.camadas:
            y = camada.propagar(y)
        return y
    
    def delta_saida(self, yN, y):
        """
        𝒚 Vector de treino de saída
        𝒚^𝑁 Vector de saída gerado pela rede
        𝒚 Dimensão do vector de treino de saída
        
        1. delta_saída 𝒚^𝑁, 𝒚 :
            2. return [ 𝑦𝑘^𝑁 − 𝑦𝑘 for 𝑘 = 1 𝑡𝑜 𝒚 ]
        """
        return [ykN - yk for ykN, yk in zip(yN, y)]

    def retropropagar(self, deltaN, alpha, beta):
        """
        𝜹^𝑁    Vector de variação de erro de saída da rede
        𝜹^𝑛    Vector de variação de erro de saída para cada camada n
        𝛿𝑗      Componente j do vector de variação de erro de saída para cada camada
        𝛼      Taxa de aprendizagem
        𝛽      Factor de momento
        
        1. retropropagar(𝜹^𝑁, 𝛼, 𝛽):
            2. 𝜹^𝑛 = 𝜹^𝑁
            3. for 𝑛 = 𝑁 to 2 step − 1:             Camada de entrada apenas mantém entradas
                4. 𝒚^𝑛−1 = 𝑐𝑎𝑚𝑎𝑑𝑎𝑠 [𝑛 − 1]. 𝒚
                5. 𝑑^𝑛−1 = 𝑐𝑎𝑚𝑎𝑑𝑎𝑠 [𝑛 − 1]. 𝑑𝑠       Dimensão da camada n - 1
                6. 𝑑^𝑛 = 𝑐𝑎𝑚𝑎𝑑𝑎𝑠 [𝑛]. 𝑑𝑠             Dimensão da camada n
                7. 𝑛𝑒𝑢𝑟𝑛 = 𝑐𝑎𝑚𝑎𝑑𝑎𝑠 [𝑛]. 𝑛𝑒𝑢𝑟𝑜𝑛𝑖𝑜𝑠     Neurónios da camada n
                8. 𝜹^𝑛−1 = [σ𝑗=1 𝑑^𝑛 𝑛𝑒𝑢𝑟𝑛 𝑗 . 𝑤𝑖 ∗ 𝛿𝑗^𝑛 ∗ 𝑛𝑒𝑢𝑟𝑛 𝑗 . 𝑦′ for 𝑖 = 1 to 𝑑^𝑛−1]
                9. 𝑐𝑎𝑚𝑎𝑑𝑎𝑠[𝑛]. adaptar(𝜹^𝑛, 𝒚^𝑛−1, 𝛼, 𝛽)
                10. 𝜹^𝑛 = 𝜹^𝑛−1
        """
        delta = deltaN
        for n in range(self.N - 1, 0, -1):  
            y_anterior = self.camadas[n - 1].y
            neur = self.camadas[n].neuronios
            d_anterior = self.camadas[n - 1].ds

            delta_anterior = [
                sum(neur[j].w[i] * delta[j] * neur[j].dy for j in range(len(neur)))
                for i in range(d_anterior)
            ]

            self.camadas[n].adaptar(delta, y_anterior, alpha, beta)
            delta = delta_anterior

    def adaptar(self, x, y, alpha, beta):
        """
        𝒙 Vector de treino de entrada
        𝒚 Vector de treino de saída
        𝒚^𝑁 Vector de saída gerado pela rede
        𝜹^𝑁 Vector de variação de erro de saída
        𝜀 Erro médio de saída da rede
        𝐾 Dimensão do vector de perda
        𝛼 Taxa de aprendizagem
        𝛽 Factor de momentoo
        
        1. adaptar(𝒙, 𝒚, 𝛼, 𝛽):
            2. 𝒚^𝑁 = propagar(𝒙)
            3. 𝜹^𝑁 = delta_saída(𝒚^𝑁, 𝒚)
            4. retropropagar(𝜹^𝑁, 𝛼, 𝛽)
            5. K = |𝜹^𝑁|
            6. 𝜀 = 1/𝐾 σ𝐾𝑘=1(𝛿𝑘^𝑁)^𝟐
            7. return 𝜀
        """
        yN = self.propagar(x)
        deltaN = self.delta_saida(yN, y)
        self.retropropagar(deltaN, alpha, beta)
        K = len(deltaN)
        epsilon = sum(dk ** 2 for dk in deltaN) / K
        return epsilon

    def treinar(self, X, Y, nepocas, epsilon_max, alpha, beta):
        """
        𝑿 Conjunto de entradas 𝒙
        𝒀 Conjunto de saídas 𝒚
        𝑛é𝑝𝑜𝑐𝑎𝑠 Número de épocas de treino
        𝜀𝑚𝑎𝑥 Erro máximo
        𝛼 Taxa de aprendizagem
        𝛽 Factor de momento

        1. treinar(𝑿, 𝒀, 𝑛é𝑝𝑜𝑐𝑎𝑠, 𝜀𝑚𝑎𝑥, 𝛼, 𝛽):
            2. for 𝑛é𝑝𝑜𝑐𝑎𝑠:
            3. 𝜀 = 0                        𝜀 - erro de treino por época
            4. for 𝒙 ∈ 𝑿, 𝒚 ∈ 𝒀:
                5. 𝜀𝒙 = adaptar(𝒙, 𝒚, 𝛼, 𝛽)        𝜀𝒙 - erro de treino por entrada 𝒙
                6. 𝜀 = max(𝜀, 𝜀𝒙):
            7. if 𝜀 ≤ 𝜀𝑚𝑎𝑥:
                8. break
        """
        for epoca in range(nepocas):
            epsilon = 0
            for x, y in zip(X, Y):
                epsilon_x = self.adaptar(x, y, alpha, beta)
                epsilon = max(epsilon, epsilon_x)
            if epsilon <= epsilon_max:
                print(f"Convergiu na época {epoca + 1} com erro {epsilon:.6f}")
                break
        return epsilon
             
    # Apenas para efeitos de teste, para definir pesos e pendor interno da rede
    def definir_pesos(self, pesos):
        """
        pesos: lista de (W, b) por camada densa, por ordem
        """
        camadas_densas = [c for c in self.camadas if isinstance(c, CamadaDensa)]
        for camada, (W, b) in zip(camadas_densas, pesos):
            for neuronio, w_linha, bi in zip(camada.neuronios, W, b):
                neuronio.w = w_linha
                neuronio.b = bi