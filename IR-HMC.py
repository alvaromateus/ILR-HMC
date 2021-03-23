import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import array as arr
from scipy import stats
import math
from datetime import datetime

# Parametrizações iniciais
metodo = 0 # 0-Regressão Linar   |   1-Regressão Polinomial
grau_polinomio = 3

#nome_base = "Teste"
#df = pd.read_csv("datasets_GO/cellcycle_GO/cellcycle_GO.train.test.csv")
#df_hierarquia = pd.read_csv("datasets_GO/cellcycle_GO/hierarquia2.csv")

#nome_base = "Cellcycle"
#df = pd.read_csv("datasets_GO/cellcycle_GO/GOCellcycle(atributosFaltando).csv")
#df_hierarquia = pd.read_csv("datasets_GO/cellcycle_GO/hierarquia_cellcycle.csv")
#arquivo_divisao = "Helyane/GOCellcycleInstanciasTreinamento.txt"

nome_base = "Church"
df = pd.read_csv("Helyane/GOChurch(atributosFaltando).txt")
df_hierarquia = pd.read_csv("datasets_GO/church_GO/hierarquia_church.csv")
arquivo_divisao = "Helyane/GOChurchInstanciasTreinamento.txt"

#nome_base = "Eisen"
#df = pd.read_csv("Helyane/GOEisen(atributosFaltando).txt")
#df_hierarquia = pd.read_csv("datasets_GO/eisen_GO/hierarquia.csv")
#arquivo_divisao = "Helyane/GOEisenInstanciasTreinamento.txt"

#nome_base = "Expr"
#df = pd.read_csv("Helyane/GOExpr(atributosFaltando).txt")
#df_hierarquia = pd.read_csv("datasets_GO/expr_GO/hierarquia.csv")
#arquivo_divisao = "Helyane/GOExprInstanciasTreinamento.txt"

#nome_base = "Gasch1"
#df = pd.read_csv("Helyane/GOGasch1(atributosFaltando).txt")
#df_hierarquia = pd.read_csv("datasets_GO/gasch1_GO/hierarquia.csv")
#arquivo_divisao = "Helyane/GOGasch1InstanciasTreinamento.txt"

#nome_base = "Gasch2"
#df = pd.read_csv("Helyane/GOGasch2(atributosFaltando).txt")
#df_hierarquia = pd.read_csv("datasets_GO/gasch2_GO/hierarquia.csv")
#arquivo_divisao = "Helyane/GOGasch2InstanciasTreinamento.txt"

#nome_base = "Seq"
#df = pd.read_csv("Helyane/GOSeq(atributosFaltando).txt")
#df_hierarquia = pd.read_csv("datasets_GO/seq_GO/hierarquia.csv")
#arquivo_divisao = "Helyane/GOSeqInstanciasTreinamento.txt"

#nome_base = "Spo"
#df = pd.read_csv("Helyane/GOSpo(atributosFaltando).txt")
#df_hierarquia = pd.read_csv("datasets_GO/spo_GO/hierarquia.csv")
#arquivo_divisao = "Helyane/GOSpoInstanciasTreinamento.txt"

def retorna_ascendentes(conjunto, classe):  # retorna classe pai
    pais = []  # variável que será retornada do tipo array pois será possível haver mais de um ascendente
    conjunto = conjunto[conjunto['item'].str.contains(classe)]  # filtra somente os registros com a classe solicitada

    for index, row in conjunto.iterrows():  # Percorre todos os registros
        pais.append(
            row[0].split("/")[0])  # caso encontre adiciona a variável que será retornada. Utiliza o divisor "/"
    return pais

def verifica_rotulos_ascendentes(base, rotulo, base_hierarquia):  # Verificaçãose há outros exemplos com rótulo ascendnete(Verificação hierárquica)
    rotulos = []  # variável que será retornada
    # Caso o rótulo ascendente não possuir exemplos será acionado o próprio método recursivamente
    if (base[base['class'].str.contains(rotulo)].empty):
        return (verifica_rotulos_ascendentes(base, retorna_ascendentes(base_hierarquia, rotulo)[0],base_hierarquia))
    else:  # Caso o  rótulo ascendente possuir exemplos este é adicionado a variável de retorno
        return (base[base['class'].str.contains(rotulo)])

def correlacao(conjunto, indice_atributo_a, indice_atributo_b, metodo, grau):
    x = conjunto.iloc[:, indice_atributo_a].values
    y = conjunto.iloc[:, indice_atributo_b].values    

    if (metodo == 0):        
        x = x.reshape(-1, 1)
        modelo = LinearRegression()
        modelo.fit(x, y)
        indice = modelo.score(x, y)
        if indice == 1:
            return 0
        else:
            return indice
    if (metodo == 1):
        x = x.reshape(-1, 1)
        lin_reg_2 = LinearRegression()

        poly_reg = PolynomialFeatures(degree=grau)
        x_poly = poly_reg.fit_transform(x)                

        lin_reg_2.fit(x_poly,y)
        
        indice = lin_reg_2.score(x_poly, y)
        if indice == 1:
            return 0
        else:
            return indice

def isNaN(num):
    return num != num

def verificacao_hierarquica_multirrotulo(base, rotulos_dado_faltante, base_hierarquia):
    conjunto = []
    classes_encontradas = []
    classes_nao_encontradas = []

    # Verificação Multirrótulo
    for i in rotulos_dado_faltante:
        conjuntos_semelhantes = base[base.iloc[:, base.columns.get_loc('class')].str.contains(i)]
        if not (conjuntos_semelhantes.empty):
            conjunto.append(conjuntos_semelhantes)
            classes_encontradas.append(i)  # adiciona o item a variável de control
        else:
            classes_nao_encontradas.append(i)
    #print(len(conjunto))

    if (len(conjunto) > 1):  # une dataframes
        conjunto_ = pd.merge_ordered(conjunto[0], conjunto[1], fill_method="ffill")
        contador = 2
        while len(conjunto) > contador:
            conjunto_ = pd.merge_ordered(conjunto_, conjunto[contador])
            contador = contador+1
        conjunto = conjunto_
    else:
        conjunto = conjunto[0]

    # Verificação Hierárquica
    dados_rotulos = []
    lista_final = list(set(classes_nao_encontradas) - set(classes_encontradas))  # retorna a diferença das duas listas
    if ((len(lista_final)) != 0):
        if (conjunto[conjunto.iloc[:, conjunto.columns.get_loc('class')].str.contains(lista_final[0])].empty):
            ascendentes = retorna_ascendentes(base_hierarquia, lista_final[0])
            for j in ascendentes:  # verificação caso seja estrutura do tipo DAG (vários ascendentes / Pais)
                dados_rotulos.append(verifica_rotulos_ascendentes(base, j, base_hierarquia))  # adiciona os rótulos a variável de retorno
            if len(dados_rotulos)>1:
                conjunto_ = pd.merge_ordered(dados_rotulos[0], dados_rotulos[1], fill_method="ffill")
                contador = 2
                conjunto_ = conjunto_.drop_duplicates()
                if len(dados_rotulos)>2:
                    while len(dados_rotulos) > contador:
                        conjunto_ = pd.merge_ordered(conjunto_, dados_rotulos[contador], fill_method="ffill")
                        conjunto_ = conjunto_.drop_duplicates()
                        contador = contador + 1
                else:
                    conjunto = pd.merge_ordered(conjunto, conjunto_, fill_method="ffill")
            else:
                conjunto = pd.merge_ordered(conjunto, dados_rotulos[0], fill_method="ffill")
    return conjunto.fillna(conjunto.mean(0)) # caso o conjunto resultante possua dados faltantes faz imputação pela média

def melhor_correlacao(indice_atributo, conjunto, metodo, grau, exemplo_imputado):
    contador = 0
    correlacoes = []
    colunas = []
    while (contador < conjunto.iloc[0, :].size - 1):
        correlacoes.append(correlacao(conjunto, indice_atributo, contador, metodo, grau))
        colunas.append(conjunto.columns[contador])
        contador = contador + 1
        
    correlacoes = pd.DataFrame(np.array(correlacoes).reshape(conjunto.iloc[0, :].size-1, 1), columns=list('c'))
    colunas = pd.DataFrame(np.array(colunas).reshape(conjunto.iloc[0, :].size-1, 1), columns=list("a"))

    correlacoes = correlacoes.join(colunas, lsuffix='coeficiente', rsuffix='atributo')
    correlacoes.columns = ['coeficiente', 'atributo']
    correlacoes = correlacoes.sort_values(by='coeficiente', ascending=False).reset_index(drop=True)

    # caso a coluna com melhor correlação tenha dado faltante no exemplo em verificado, a coluna é removida e é realizada nova verificação
    if (isNaN(exemplo_imputado[correlacoes.loc[0:0]['atributo'].values[0]])): 
        conjunto = conjunto.drop(columns=[correlacoes.loc[0:0]['atributo'].values[0]])
        return melhor_correlacao(indice_atributo,conjunto,metodo,grau,exemplo_imputado)
    else:
        return correlacoes.loc[0:0]['atributo'].values[0],correlacoes.loc[0:0]['coeficiente'].values[0]

def regressao_polinomial(coluna_melhor_correlacao, indice_atributo, conjunto, variavel_independente, grau):
    coluna_melhor_correlacao = conjunto.columns.get_loc(coluna_melhor_correlacao)

    x = conjunto.fillna(0).iloc[:, indice_atributo].values
    y = conjunto.fillna(0).iloc[:, coluna_melhor_correlacao].values

    poly_reg = PolynomialFeatures(degree=grau)

    x = x.reshape(-1, 1)

    x_poly = poly_reg.fit_transform(x)
    poly_reg.fit(x_poly,y)
    #modelo.fit(x_polinomio, y)    

    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(x_poly,y)
    resultado = lin_reg_2.predict(poly_reg.fit_transform([[variavel_independente]]))[0]

    return resultado

def regressao_linear(coluna_melhor_correlacao, indice_atributo, conjunto, variavel_independente):
    coluna_melhor_correlacao = conjunto.columns.get_loc(coluna_melhor_correlacao)

    x = conjunto.fillna(0).iloc[:, indice_atributo].values
    y = conjunto.fillna(0).iloc[:, coluna_melhor_correlacao].values

    x = x.reshape(-1, 1)
    modelo = LinearRegression()
    modelo.fit(x, y)

    return modelo.predict([[variavel_independente]])[0]

def normalize(df_input):
    base_drop = pd.DataFrame(df_input.drop(columns=['class'])) # Retira coluna da classe para fazer normalização
    coluna_classe = pd.DataFrame(df_input['class'])

    x = base_drop.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x) # Normaliza com Min/Max
    df_input = pd.DataFrame(x_scaled).round(3)
    df_input = df_input.join(coluna_classe, lsuffix='_caller', rsuffix='_other') # Adiciona coluna classe novamente

    #df.to_csv(r'//home/alvaro/Estudo/Clus-HMC/data/church_faltantes/GOChurch(atributosFaltando).arff', index=False, header=True)
    return df_input
    #df.to_csv(r'normalizado.csv', index=False, header=True)

def any_nan(conjunto):
    nan = 0
    not_nan = 0
    for x in conjunto:
        if isNaN(x):
            nan = nan+1
        else:
            not_nan = not_nan+1
    #print(nan)
    #print(not_nan)
    
    if not_nan>0:
        return False
    else:
        return True           

def undo_split(arquivo_instancias_treinamento, base_imputada):
    instancias_treinamento = pd.read_csv(arquivo_instancias_treinamento, header=None)
    #base_imputada = pd.read_csv("normalizado.csv")

    treinamento = base_imputada.loc[base_imputada.index[instancias_treinamento[0]-1]]
    teste = base_imputada.loc[base_imputada.index.difference(instancias_treinamento[0]-1)]
    return teste,treinamento
    
#################################### MAIN ##############################################

coluna = 0
quantidade_regressao = 0
quantidade_media = 0

while (coluna < df.iloc[0, :].size-1):
    linha = 0
    while(linha < df['class'].size):
        if isNaN(df.loc[linha][coluna]): # verifica se a linha e coluna possui dado faltante
            rotulos_dado_faltante = df.loc[linha][df.columns.get_loc('class')].split('@')  # seleciona linha x coluna
            exemplo_imputado = df.loc[linha]  # armazena o exemplo a ser imputado 
            df_ = df.drop(linha)  # cria copia removendo exemplo com dado faltante do conjunto

            conjunto = verificacao_hierarquica_multirrotulo(df_, rotulos_dado_faltante, df_hierarquia)  # define conjunto semelhante
            print("Linha: "+str(linha)+" | Coluna: "+str(coluna) + " | "+df.columns.values[coluna] +" | Time:" +datetime.now().strftime("%d/%m/%Y %H:%M:%S"))            
            # define atributos categórios das bases de dados
            atributos_categoricos = (['spo_failed_pcr','spo_blast_homology_within_genome','spo_overlaps_another_orf','failed_pcr','blast_homology_within_genome','church_chip_affymetrix_chip'])            
            
            if not (df.columns.values[coluna] in atributos_categoricos):
                # define atributo com melhor correlação
                atributo_melhor_correlacao = melhor_correlacao(coluna, conjunto, metodo, grau_polinomio, exemplo_imputado)  
                if (atributo_melhor_correlacao[1] >= 0.3): # Se correlação for boa (maior ou igual que 0.3) usa regressão
                    print("REGRESSÃO --------")
                    print(atributo_melhor_correlacao[1])
                    variavel_independente = df[atributo_melhor_correlacao[0]].loc[linha]                    
                    if (metodo == 0): # regressão linear
                        regressao = regressao_linear(atributo_melhor_correlacao[0], coluna, conjunto.drop(columns=['class']), variavel_independente)
                        df.iloc[linha, coluna] = round(regressao,2)
                    elif (metodo == 1): # regressão polinomial
                        regressao = regressao_polinomial(atributo_melhor_correlacao[0], coluna, conjunto.drop(columns=['class']), variavel_independente, grau_polinomio)
                        df.iloc[linha, coluna] = round(regressao,2)        
                    quantidade_regressao = quantidade_regressao+1                                  
                else: # se correlação não for boa usa média (<0.3)
                    print("MEDIA --------")
                    print(atributo_melhor_correlacao[1])
                    media = conjunto.iloc[:, coluna].mean()
                    if not (isNaN(media)):
                        df.iloc[linha, coluna] = media # caso não exista como verificar a média
                    else:
                        df.iloc[linha, coluna] = 0
                    quantidade_media = quantidade_media+1
            else:
                print("Atributo categórico!")
                #variavel_independente = df[atributo_melhor_correlacao[0]].mode()[0]
        linha = linha + 1        
    coluna = coluna + 1
    print("Coluna concluída")        
print(df)
#normalizado = normalize(df) # normaliza dados
#teste, treinamento = undo_split(arquivo_divisao, normalizado) # divide em trteinamento e teste
#treinamento.to_csv(r'treinamento-'+nome_base+"-M"+str(quantidade_media)+"-R"+str(quantidade_regressao)+("-linear"if metodo==0 else "-polinomial_"+str(grau_polinomio))+'.csv', index=False, header=True) # salva nos arquivos correspondenttes
#teste.to_csv(r'teste-'+nome_base+"-M"+str(quantidade_media)+"-R"+str(quantidade_regressao)+("-linear"if metodo==0 else "-polinomial_"+str(grau_polinomio))+'.csv', index=False, header=True)