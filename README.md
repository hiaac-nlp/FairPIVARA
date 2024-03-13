# FairPIVARA

## main.py
Utilizado para os resultados finais do trabalho, cria as duas tabelas de resultados. Pode utilizar as remoções de dimensões propostas através da utilização do FairPIVARA.py
-   Classification: Realiza a classificação no conjunto de textos "bons" e "ruins". Através do modelo escolhido, realiza-se uma classificação no conjunto de dados de bias, mensurando quantos valores foram dados como bons  ou ruins, através de quantidade e pontuação.
-   Comparison: Comparação entre dois conceitos e a pontuação final. O quanto cada conceito se sobrepõe ao outro. OBS: Utiliza a métrica do artigo, caliskan_test.

## FairPIVARA.py

-   Calculate_bias_separately: Calcula o bias total e quais as dimensões devem ser removidas para alcançar o bias mínimo. O Calculo é realizado por conceito.
-   Calculate_bias_together: Calcula o bias total e quais as dimensões devem ser removidas para alcançar o bias mínimo. O Calculo é realizado em conjunto.
-   bias_calculation: Comparação entre dois conceitos e a pontuação final. O quanto cada conceito se sobrepõe ao outro. OBS: Utiliza a métrica do artigo, caliskan_test.

## ZeroShotClassification.ipynb
-   Notebook para o calculo da classificação zeroshot em modelos clip para os conjuntos de dados selecionados.


OBS: Diferença entre main.py:Comparison e FairPIVARA.py:bias_calculation??