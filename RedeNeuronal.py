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
#   phi   — função de activação das camadas densas.
#   pesos — opcional: lista de (W, b) por camada densa para substituir os
#            pesos aleatórios. Usado para replicar os exemplos dos slides.
#
# propagar(x)
#   Executa o forward pass: y é inicializado com x e transformado sequencialmente
#   por cada camada. A saída final y^N é a resposta da rede para a entrada x.
#
# prever(X)
#   Aplica propagar(x) a cada amostra x do conjunto X (inferência em lote).
#   Retorna Y — lista de respostas correspondentes.
# 
# definir_pesos(pesos) 
#   Método auxiliar de validação e testes: 
#   Permite substituir os pesos e pendores gerados aleatoriamente por matrizes 
#   e vetores de pesos (W, b), fornecidos manualmente para cada camada densa.
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 1.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 2.pdf, ISEL 2026
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
    
    ## CORRIGIR COMENTARIOS DE FUNÇÕES DE ADAPTAR, RETROPROPAGAR E TREINAR, PARA ESTAR EM CONFORMIDADE COM O QUE ESTÁ IMPLEMENTADO
    def delta_saida(self, yN, y):
        """
        1. delta_saida(y^N, y):
            2. return [y_k^N - y_k for k = 1 to |y|]

        yN — vector de saída gerado pela rede
        y  — vector de treino de saída (alvo)
        """
        return [ykN - yk for ykN, yk in zip(yN, y)]

    def retropropagar(self, deltaN, alpha):
        """
        1. retropropagar(delta^N, alpha):
            2. delta^n = delta^N
            3. for n = N to 2 step -1:
                4. y^(n-1) = camadas[n-1].y
                5. delta^(n-1) = [sum(neur[j].w[i] * delta_j * neur[j].y' for j) for i]
                6. camadas[n].adaptar(delta^n, y^(n-1), alpha)
                7. delta^n = delta^(n-1)

        deltaN — vector de variação de erro de saída da rede
        alpha  — taxa de aprendizagem
        """
        delta = deltaN
        for n in range(self.N - 1, 0, -1):  # N-1 até 1 (exclui camada de entrada)
            y_anterior = self.camadas[n - 1].y
            neur = self.camadas[n].neuronios
            d_anterior = self.camadas[n - 1].ds

            # delta^(n-1) para cada i da camada anterior
            delta_anterior = [
                sum(neur[j].w[i] * delta[j] * neur[j].dy for j in range(len(neur)))
                for i in range(d_anterior)
            ]

            self.camadas[n].adaptar(delta, y_anterior, alpha)
            delta = delta_anterior

    def adaptar(self, x, y, alpha):
        """
        1. adaptar(x, y, alpha):
            2. y^N = propagar(x)
            3. delta^N = delta_saida(y^N, y)
            4. retropropagar(delta^N, alpha)
            5. epsilon = sum(delta_k^N²) / K
            6. return epsilon

        x     — vector de treino de entrada
        y     — vector de treino de saída
        alpha — taxa de aprendizagem
        """
        yN = self.propagar(x)
        deltaN = self.delta_saida(yN, y)
        self.retropropagar(deltaN, alpha)
        K = len(deltaN)
        epsilon = sum(dk ** 2 for dk in deltaN) / K
        return epsilon

    def treinar(self, X, Y, nepocas, epsilon_max, alpha):
        """
        1. treinar(X, Y, nepocas, epsilon_max, alpha):
            2. for nepocas:
                3. epsilon = 0
                4. for x in X, y in Y:
                    5. epsilon_x = adaptar(x, y, alpha)
                    6. epsilon = max(epsilon, epsilon_x)
                7. if epsilon <= epsilon_max:
                    8. break

        X          — conjunto de entradas
        Y          — conjunto de saídas alvo
        nepocas    — número máximo de épocas
        epsilon_max — erro máximo para paragem antecipada
        alpha      — taxa de aprendizagem
        """
        for epoca in range(nepocas):
            epsilon = 0
            for x, y in zip(X, Y):
                epsilon_x = self.adaptar(x, y, alpha)
                epsilon = max(epsilon, epsilon_x)
            if epsilon <= epsilon_max:
                print(f"Convergiu na época {epoca + 1} com erro {epsilon:.6f}")
                break
        return epsilon