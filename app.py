import os 
import streamlit as st
#import pandas as pd
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score

#--------------------------------------------------------
from autogluon.tabular import  TabularDataset, TabularPredictor 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import collections
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, roc_curve, average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score, matthews_corrcoef, auc)
#--------------------------------------------------------

os.system('clear')

st.set_page_config(
    page_title="Previsão de Vulnerabilidade da Mulher à Violência Doméstica",
    layout="wide"
)

@st.cache_data
def load_data_and_model():
    carros = pd.read_csv("car.csv",sep=",")
    encoder = OrdinalEncoder()

    for col in carros.columns.drop('class'):
        carros[col] = carros[col].astype('category')

    X_encoded = encoder.fit_transform(carros.drop('class',axis=1))

    y = carros['class'].astype('category').cat.codes

    X_train,X_test, y_train, y_test = train_test_split(X_encoded,y, test_size=0.3, random_state=42)

    modelo = CategoricalNB()
    modelo.fit(X_train,y_train)

    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)

    return encoder, modelo, acuracia, carros

encoder, modelo, acuracia, carros = load_data_and_model()

def load_pesquisa():
    # carrega o arquivo do drive
    dataset_file = 'dataset_tabular.csv'
    df = pd.read_csv(dataset_file)

    dataset_arq = 'Dataset_ViolenciaMulher_Completo.csv'
    df_arq = pd.read_csv(dataset_arq, encoding="utf-16", sep=',')

    return df, df_arq 

df, df_arq = load_pesquisa()

# remove colunas não necessárias para o classificador
df.drop(columns=['TipoViolencia_codigo', 'TipoViolencia',
                 'IdadeViolencia_codigo', 'IdadeViolencia',
                 'VezesViolencia_codigo', 'VezesViolencia',
                 'AgressorViolencia_codigo', 'AgressorViolencia',
                 'MotivoViolencia_codigo', 'MotivoViolencia'], inplace=True)

#df.drop(columns=['Ano', 'Respondente', 'Regiao', 'Estado', 'EstadoCivil',
#                 'Estado_codigo', 'UF', 'Casada', 'TemFilhos', 'CorRaca',
#                 'Escolaridade', 'Ocupacao', 'ReligiaoCrenca',
#                 'RendaTotalFamilia', 'PosicionamentoPolitico',
#                 'SofreuViolencia'], inplace=True)
#
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


#input_features  = [
#       st.selectbox("Preço:",carros['buying'].unique()),
#       st.selectbox("Manutenção:",carros['maint'].unique()),
#       st.selectbox("Portas:",carros['doors'].unique()),
#       st.selectbox("Capacidade de Passegeiros:",carros['persons'].unique()),
#       st.selectbox("Porta Malas:",carros['lug_boot'].unique()),
#       st.selectbox("Segurança:",carros['safety'].unique()),
#       ]

df = df.loc[df['Regiao'] != 'NP']
df = df.loc[df['EstadoCivil'] != 'NP']
df = df.loc[df['TemFilhos'] != 'NP']
df = df.loc[df['CorRaca'] != 'NP']
df = df.loc[df['Escolaridade'] != 'NP']
df = df.loc[df['ReligiaoCrenca'] != 'NP']
df = df.loc[df['Ocupacao'] != 'NP']
df = df.loc[df['RendaTotalFamilia'] != 'NP']
df = df.loc[df['PosicionamentoPolitico'] != 'NP']

input_features = [
   st.selectbox("Região:",df['Regiao'].unique()),
   st.selectbox("Estado civil:",df['EstadoCivil'].unique()),
   st.selectbox("TemFilhos:",df['TemFilhos'].unique()),
   st.selectbox("CorRaca:",df['CorRaca'].unique()),
   st.selectbox("Escolaridade:",df['Escolaridade'].unique()),
   st.selectbox("Religiao/Crenca:",df['ReligiaoCrenca'].unique()),
   st.selectbox("Ocupacao:",df['Ocupacao'].unique()),
   st.selectbox("Renda total familia:",df['RendaTotalFamilia'].unique()),
   st.selectbox("Posicionamento político:",df['PosicionamentoPolitico'].unique()),
   ]

data = {
    'Regiao': ['Norte', 'Nordeste', 'Sudeste', 'Sul', 'Centro-Oeste'],
    'Regiao_codigo': [1, 2, 3, 4, 5],
    'EstadoCivil': ['NS/NR', 'Solteira', 'Casada','Divorciada','Viúva','Separada','União estável'],
    'EstadoCivil_codigo': [0, 1, 2, 3, 4, 5, 6],
    'TemFilhos': ['NS/NR','Sim','Não'],
    'TemFilhos_codigo': [0, 1, 2],
    'CorRaca': ['NS/NR','Branca','Preta','Parda','Indígena','Amarela'],
    'CorRaca_codigo': [0, 1, 2, 3, 4, 5],
	 'Escolaridade': ['NS/NR', 'Não Alfabetizada', 'Ensino Fundamental', 'Ensino Médio', 'Ensino Superior', 'Pós-Graduação'],
    'Escolaridade_codigo': [0, 1, 2, 3, 4, 5],
    'Ocupacao': ['NS/NR', 'Não Trabalha', 'Assalariada', 'Estudante', 'Empregada Doméstica', 'Autonoma', 'Aposentada', 'Outra'],
    'Ocupacao_codigo': [0, 1, 2, 3, 4, 5, 6, 9],	
    'ReligiaoCrenca': ['NS/NR', 'Católica', 'Evangélica', 'Espírita', 'Umbanda ou Candomblé', 'Outras', 'Sem Religião'],
    'ReligiaoCrenca_codigo': [0, 1, 2, 3, 4, 8, 9],
    'RendaTotalFamilia': ['NS/NR','Sem renda','Até 2 salário mínimos','De 3 a 5 salários mínimos','De 6 a 10 salários mínimos', 'De 11 a 20 salários mínimos', 'Mais de 21 salários mínimos'],
    'RendaTotalFamilia_codigo': [0, 1, 2, 3, 4, 5, 6],
    'PosicionamentoPolitico': ['NS/NR', 'Esquerda', 'Direita', 'Centro', 'Nenhuma das anteriores'], 
    'PosicionamentoPolitico_codigo': [0, 1, 2, 3, 4]
}

# Dicionário para mapeamento de campos para código
campo_codigo_map = {
    # Adicione os outros campos
   'Regiao': 'Regiao_codigo',
   'EstadoCivil': 'EstadoCivil_codigo',
   'CorRaca_codigo': 'CorRaca',
   'TemFilhos': 'TemFilhos_codigo',
	'Escolaridade': 'Escolaridade_codigo',
   'Ocupacao': 'Ocupacao_codigo',	
   'ReligiaoCrenca': 'ReligiaoCrenca_codigo',
   'RendaTotalFamilia': 'RendaTotalFamilia_codigo',
   'PosicionamentoPolitico': 'PosicionamentoPolitico_codigo'
}

simulacao = []
for feature in input_features:
   for chave, campo in campo_codigo_map.items():
      if feature in data[chave]: 
         posicao = data[chave].index(feature)     
         #print(f'Para {campo} respondeu {feature} na posição {posicao}')
         simulacao.append(posicao)

# Guto, você deve montar um dataframe com alinha resposta com os seguintes campos:
'''
Ano                             
Respondente                     
Regiao_codigo                   
Regiao                          
Estado_codigo                   
Estado                          
UF                              
EstadoCivil_codigo              
EstadoCivil                     
Casada_codigo                   
Casada                          
TemFilhos_codigo                
TemFilhos                       
CorRaca_codigo                  
CorRaca                         
Escolaridade_codigo             
Escolaridade                    
Ocupacao_codigo                 
Ocupacao                        
ReligiaoCrenca_codigo           
ReligiaoCrenca                  
RendaTotalFamilia_codigo        
RendaTotalFamilia               
PosicionamentoPolitico_codigo   
PosicionamentoPolitico          
SofreuViolencia_codigo          
SofreuViolencia                 

'''
print('Simulação array')
print(simulacao)
# Dicionário para mapeamento de campos para código


if st.button("Processar"):
  indice = 0
  for feature_selected in input_features:
    print('feature_selected', feature_selected, indice)
    indice += 1
    st.write(feature_selected)


#input_df = pd.DataFrame([codigo_regiao, ])

# print(input_features_codes)
# Filtrar o DataFrame para encontrar o código correspondente a 'Solteiro' na coluna 'estado civil'
#codigo_solteiro = df.loc[df['estado civil'] == 'Solteiro', 'estado civil_codigo'].unique()

# Exibir o resultado
#print("Código para 'Solteiro':", codigo_solteiro[0] if len(codigo_solteiro) > 0 else "Não encontrado")



# remove resultados diferentes de 'Sim' ou 'Não'
print('Dataset size (complete):', len(df))
df = df[df['SofreuViolencia_codigo'].isin([1, 2])]
print('Dataset size (only: yes/no):', len(df))


# Substitui o valor do Não (2) para (0)
df.replace(2, 0, inplace=True)

st.title("Previsão de Vulnerabilidade da Mulher à Violência Doméstica")
st.write(f"Acurácia do modelo: {acuracia:.2f}")

st.dataframe(df)

# cria o dataset para processamento pelo AutoGluon
dataset = TabularDataset(df)
dataset.head()

st.dataframe(dataset)

dataset.info()

print('Dataset')
print(dataset)

target = 'SofreuViolencia_codigo'
dataset[target].describe()

# Separação dos Conjuntos da Treino e Testes

train_size = int(len(dataset) * 0.8) # 80% do conjunto completo
seed = 2024
# train / test split
dataset_train = dataset.sample(train_size, random_state=seed)
dataset_test  = dataset.drop(dataset_train.index)

print('dataset_train')
print(dataset_train)

print('dataset_test')
print(dataset_test)

print(dataset_test.iloc[0])

print('Chegou aqui')
y_test = dataset_test[target]

print(f'Train size: {len(dataset_train)}')
print(f'Test size : {len(dataset_test)}')

st.write(f'Train size: {len(dataset_train)}')
st.write(f'Test size : {len(dataset_test)}')

'''
Treinamento

É criado agora um TabularPredictor através da especificação do nome da coluna e então treinado a partir do dataset. AutoGluon deve reconhecer que pode ser uma tarefa de classificação, executar a engenharia de características automaticamente, treinar os múltiplos modelos e então combinar os modelos para criar um preditor ao final.
'''

predictor = TabularPredictor(label=target) # eval_metric='accuracy
#predictor = TabularPredictor(label=target, eval_metric='f1')
predictor.fit(dataset_train)

'''
Predição

Uma vez que temos o preditor que está ajustado ao **dateset de treino**, pode-se carregar um **dataset de teste** separado para ser usado para predição e avaliação de resultados.
'''

y_pred = predictor.predict(dataset_test.drop(columns=[target]))
y_pred.describe()

'''
Avaliação

O preditor pode ser avaliado com base no dataset de testes usando a função `evaluate()`, que mede o quanto o preditor criado está ajustado e consegue generalizar para dados não presentes no dataset de treinamento.
'''

predictor.evaluate(dataset_test, silent=True)

'''
O `TabularPredictor` do AutoGluon também provê a função `leaderboard()`, que permite avaliar a performance de cada um dos modelos individuais treinados no dataset de teste.
'''
predictor.leaderboard(dataset_test)

# imprimindo o  melhor modelo
predictor.model_best
st.write(f'Imprimindo o  melhor modelo: {predictor.model_best}')

# pegando uma amostra dos dados pra testar a predição
test_sample = df.head(1)
test_sample['SofreuViolencia_codigo']

st.write(f'Predizendo com base em um exemplo: {predictor.predict(test_sample)}')
# solicitando para o preditor prever para a amostra selecionada
predictor.predict(test_sample)

summary = predictor.fit_summary()
summ_file = 'dataset_fit_summary.txt'
with open(summ_file, 'w') as f:
  print(summary, file=f)

# Curva ROC

def plot_roc_curve(y_true, y_score, is_single_fig=False):
  """
  Plot ROC Curve and show AUROC score
  """
  fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
  roc_auc = auc(fpr, tpr)
  plt.title('AUROC = {:.4f}'.format(roc_auc))
  plt.plot(fpr, tpr, 'b')
  plt.plot([0,1], [0,1], 'r--')
  plt.xlim([-0.05,1.05])
  plt.ylim([-0.05,1.05])
  plt.ylabel('TPR(True Positive Rate)')
  plt.xlabel('FPR(False Positive Rate)')
  if is_single_fig:
    plt.show()

def plot_pr_curve(y_true, y_score, is_single_fig=False):
  """
  Plot Precision Recall Curve and show AUPRC score
  """
  prec, rec, thresh = precision_recall_curve(y_true, y_score, pos_label=1)
  avg_prec = average_precision_score(y_true, y_score, pos_label=1)
  plt.title('AUPRC = {:.4f}'.format(avg_prec))
  plt.step(rec, prec, color='b', alpha=0.2, where='post')
  plt.fill_between(rec, prec, step='post', alpha=0.2, color='b')
  plt.plot(rec, prec, 'b')
  plt.xlim([-0.05,1.05])
  plt.ylim([-0.05,1.05])
  plt.ylabel('Precision')
  plt.xlabel('Recall')
  if is_single_fig:
    plt.show()

def plot_conf_mtx(y_true, y_score, thresh=0.5, class_labels=['Sim','Não'], is_single_fig=False):
  """
  Plot Confusion matrix
  """
  y_pred = np.where(y_score >= thresh, 1, 0)
  print("confusion matrix (cutoff={})".format(thresh))
  print(classification_report(y_true, y_pred, target_names=class_labels))
  conf_mtx = confusion_matrix(y_true, y_pred)
  sns.heatmap(conf_mtx, xticklabels=class_labels, yticklabels=class_labels, annot=True, fmt='d')
  plt.title('Confusion Matrix')
  plt.ylabel('True Class')
  plt.xlabel('Predicted Class')
  if is_single_fig:
    plt.show()

y_prob = predictor.predict_proba(dataset_test)
y_prob = y_prob.iloc[:,1]


fig = plt.figure(figsize=(14,4))
plt.subplot(1,3,1)
plot_roc_curve(y_test, y_prob)
plt.subplot(1,3,2)
plot_pr_curve(y_test, y_prob)
plt.subplot(1,3,3)
plot_conf_mtx(y_test, y_prob, 0.5)
eval_file = 'dataset_eval.png'
plt.savefig(eval_file)
plt.close(fig)


#if st.button("Processar"):
#    input_df = pd.DataFrame([input_features], columns=carros.columns.drop('class'))
#    input_encoded = encoder.transform(input_df)
#    predict_encoded = modelo.predict(input_encoded)
#    previsao = carros['class'].astype('category').cat.categories[predict_encoded][0]
#    st.header(f"Resultado da previsão:  {previsao}")


