Sistema inteligente para previsão de adesão a título de capitalização

A preparação inicial dos dados, foi realizada em Excel, a coluna 'y' foi convertida para binário, logo após
foram concatenados os conteúdos das colunas e separados por vírgulas.
  
O tratamento dos dados e os módulos estão detalhados e funcionais no arquivo 'Induction'.

O algoritmo de Regressão logística foi escolhido porque é uma técnica que tem muitas vantagens perante a regressão linear, 
principalmente no que diz respeito à normalidade e linearidade. Não há a necessidade da relação linear entre variável 
resposta e variáveis explicativas. Além disso, os resíduos não precisam estar normalmente distribuídos.

Ao utilizarmos este modelo, estamos interessados na probabilidade como saída. No nosso caso de títulos de capitalização, 
estaríamos interessados em obter a probabilidade de um indivíduo adquirir ou não. Para tal utilizamos a função logística, 
que resulta numa curva num formato de um S.

O teste de acurácia foi realizado sobre uma amostra de 30% da base de dados, comparando os dados previstos com os reais.

O módulo 'Inference' deve classificar uma nova instância inserida no 'new-instance'

A linguagem de programação Python foi escolhida, porque é uma linguagem de uso geral que pode fazer 
praticamente qualquer coisa como: coleta de dados, engenharia de dados, análise e muito mais. É mais simples de dominar, 
é mais fácil escrever em grande escala e com código robusto.