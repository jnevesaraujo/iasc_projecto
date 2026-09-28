###################################################################################
## Classe CamadaEntrada
###################################################################################
# Na analogia biológica, a camada de entrada corresponde às dendrites dos
# neurónios: recebe os sinais do exterior e disponibiliza-os para processamento.
#
# É a Camada 1 da rede (n=1). Ao contrário das camadas densas, não possui
# pesos nem pendor; os seus neurónios implementam a função identidade f(xi) = xi,
# ou seja, a entrada passa sem transformação.
#
# A saída y1 desta camada torna-se a entrada x2 da primeira camada escondida,
# iniciando o processo de activação directa da rede (forward pass).
#
# __init__(ds)
#   ds — dimensão de saída: número de componentes do vector de entrada x.
#   y  — vector de saída, inicializado a zeros, dimensão ds.
#
# propagar(x)
#   Implementa y = x (função identidade).
#   Não aplica soma de activação h nem função de activação phi.
#
# Fontes:
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 1.pdf, ISEL 2026
#   MORGADO, Luis; Redes Neuronais Artificiais - Parte 2.pdf, ISEL 2026
###################################################################################

class CamadaEntrada:
    def __init__(self, ds):
        """
        1. iniciar(𝑑𝑠):
            2. 𝑑𝑠 ∈ 𝑍+          Dimensão de saída (número de saídas da camada)
            3. 𝒚 = [0 for 𝑑𝑠
            ]
        """ 
        self.ds = ds
        self.y = [0 for _ in range(ds)]


    def propagar(self, x):
        """
        𝒙 - vector de entrada
        𝒚 - vector de saida 
        1. propagar (𝒙) :
            2. 𝒚 = 𝒙            Manter informação de saída
            3. return 𝒚
        """
        self.y = x
        return self.y
