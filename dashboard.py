import os 
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import plotly.express as px

os.system('clear')


st.set_page_config(
    page_title="Violência Doméstica Contra a Mulher",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")


def load_pesquisa():
    # carrega o arquivo do drive
    #dataset_file = 'dataset_tabular.csv'
    #df = pd.read_csv(dataset_file)

    dataset_arq = 'Dataset_ViolenciaMulher_Completo.csv'
    df_arq = pd.read_csv(dataset_arq, encoding="utf-16", sep=',')

    return df_arq 

df = load_pesquisa()

# remove colunas não necessárias para o classificador
df.drop(columns=[
                 'IdadeViolencia_codigo', 'IdadeViolencia',
                 'VezesViolencia_codigo', 'VezesViolencia',
                 'AgressorViolencia_codigo',
                 'MotivoViolencia_codigo', 'MotivoViolencia'], inplace=True)

#df.drop(columns=['Ano', 'Respondente', 'Regiao', 'Estado', 'EstadoCivil',
#                 'Estado_codigo', 'UF', 'Casada', 'TemFilhos', 'CorRaca',
#                 'Escolaridade', 'Ocupacao', 'ReligiaoCrenca',
#                 'RendaTotalFamilia', 'PosicionamentoPolitico',
#                 'SofreuViolencia'], inplace=True)
#
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

#df = df.loc[df['Regiao'] != 'NP']
#df = df.loc[df['EstadoCivil'] != 'NP']
#df = df.loc[df['TemFilhos'] != 'NP']
#df = df.loc[df['CorRaca'] != 'NP']
#df = df.loc[df['Escolaridade'] != 'NP']
#df = df.loc[df['ReligiaoCrenca'] != 'NP']
#df = df.loc[df['Ocupacao'] != 'NP']
#df = df.loc[df['RendaTotalFamilia'] != 'NP']
#df = df.loc[df['PosicionamentoPolitico'] != 'NP']

indicadores = {'Física': 0,
               'Moral': 0,
               'NS/NR': 0,
               'Patrimonial': 0,
               'Psicológica': 0,
               'Sexual': 0,
               'Todas as anteriores': 0}

# Ajuste dos rótulos da categoria Renda Familiar
#df = df.replace("NP", "Não Perguntado")
df = df.replace("Até 2 salário mínimos", "Até 2 salários")
df = df.replace("De 3 a 5 salários mínimos", "De 3 a 5 salários")
df = df.replace("De 6 a 10 salários mínimos", "De 6 a 10 salários")

# Ajuste dos rótulos da categoria Escolaridade
df = df.replace("Ensino Médio", "Médio")
df = df.replace("Ensino Fundamental", "Fundamental")
df = df.replace("Ensino Superior", "Superior")
df = df.replace("Não Alfabetizada", "N Alfabetizada")

# Ajuste do rótulo da categoria Ocupação
df = df.replace("Empregada Doméstica", "Emp.Doméstica")

# Agrupa categorias para aumentar clareza nos gráficos 
df = df.replace("Médio", "Até o Médio")
df = df.replace("Fundamental", "Até o Médio")
df = df.replace("De 11 a 20 salários mínimos", "Mais de 11 salários")
df = df.replace("Mais de 21 salários mínimos", "Mais de 11 salários")
df = df.replace("Branca", "Branca/Amarela")
df = df.replace("Amarela", "Branca/Amarela")
df = df.replace("Indígena", "Indíg./Parda/Preta")
df = df.replace("Parda", "Indíg./Parda/Preta")
df = df.replace("Preta", "Indíg./Parda/Preta")

with st.sidebar:
    st.title('📊 Filtro')
    st.logo('dia-nacional-de-luta-contra-violencia-a-mulher-entenda-o-ciclo-de-violencia-que-pode-levar-ao-feminicidio.jpg', size="large", link='https://i0.wp.com/surgiu.com.br/wp-content/uploads/2020/10/dia-nacional-de-luta-contra-violencia-a-mulher-entenda-o-ciclo-de-violencia-que-pode-levar-ao-feminicidio.jpg?fit=300%2C300&amp;ssl=1%22%20data-large-file=%22https://i0.wp.com/surgiu.com.br/wp-content/uploads/2020/10/dia-nacional-de-luta-contra-violencia-a-mulher-entenda-o-ciclo-de-violencia-que-pode-levar-ao-feminicidio.jpg', icon_image=None)

    # Lista os anos presentes na pesquisa e cria seletor acumulativo
    year_list = list(df.Ano.unique())[::-1] 
    selected_year = st.multiselect('Selecione um ou mais anos', year_list, year_list[0])
    
    # Indexa os dados por ano
    df_selected_year = df.loc[df['Ano'].isin(selected_year)]
    df_selected_year_sorted = df_selected_year.sort_values(by="Regiao", ascending=False)
    
    # Cria recorte dos dados por Tipo de Violência
    df_tipo_de_violencia = df_selected_year_sorted.groupby(['TipoViolencia'])['TipoViolencia_codigo'].count().reset_index()
    
    # Cria recorte dos dados por Estado
    df_violencia_estado = df_selected_year.groupby(['Estado'])['SofreuViolencia'].count().reset_index()
    df_violencia_estado = df_violencia_estado.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Renda Total Familiar
    df_violencia_renda = df_selected_year.groupby(['RendaTotalFamilia'])['SofreuViolencia'].count().reset_index()
    df_violencia_renda = df_violencia_renda.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Ocupação
    df_violencia_ocupacao = df_selected_year.groupby(['Ocupacao'])['SofreuViolencia'].count().reset_index()
    df_violencia_ocupacao = df_violencia_ocupacao.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Escolaridade
    df_violencia_escolaridade = df_selected_year.groupby(['Escolaridade'])['SofreuViolencia'].count().reset_index()
    df_violencia_escolaridade = df_violencia_escolaridade.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Regões do Brasil
    df_violencia_regiao = df_selected_year.groupby(['Regiao'])['SofreuViolencia'].count().reset_index()
    df_violencia_regiao = df_violencia_regiao.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Estado Civil
    df_violencia_estadocivil = df_selected_year.groupby(['EstadoCivil'])['SofreuViolencia'].count().reset_index()
    df_violencia_estadocivil = df_violencia_estadocivil.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Etnia
    df_violencia_etnia = df_selected_year.groupby(['CorRaca'])['SofreuViolencia'].count().reset_index()
    df_violencia_etnia = df_violencia_etnia.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Religião/Crença
    df_violencia_religiao = df_selected_year.groupby(['ReligiaoCrenca'])['SofreuViolencia'].count().reset_index()
    df_violencia_religiao = df_violencia_religiao.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Posicionamento Político
    df_violencia_politica = df_selected_year.groupby(['PosicionamentoPolitico'])['SofreuViolencia'].count().reset_index()
    df_violencia_politica = df_violencia_politica.sort_values(by="SofreuViolencia", ascending=False)

    # Cria recorte dos dados por Tipo de causador das agressões
    df_violencia_agressor = df_selected_year.groupby(['AgressorViolencia'])['SofreuViolencia'].count().reset_index()
    df_violencia_agressor = df_violencia_agressor.sort_values(by="SofreuViolencia", ascending=False)
    
    # Cria seletor para categorias secundárias que serão apresentadas em tabela
    top_selector = st.radio(
        "Top Categorias mais violentas",
        ["Região", "Estado", "Estado Civil", "Agressor", "Etnia", "Religião/Crença", "Posicionamento Político"],
    )

# Prepara layout do cabeçalho
st.title('Violência Domêstica Contra a Mulher')
st.subheader('Tipos de violência sofrida',divider=True)
kpi = st.columns((0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), gap='small')

# Transforma dataframe dos tipos de violência em dicionário
for tipo in df_tipo_de_violencia.itertuples():
    indicadores[tipo.TipoViolencia] = tipo.TipoViolencia_codigo

# Atualiza os indicadores de tipo de violência
i = 0
for indice, indicador in indicadores.items():
    kpi[i].metric(indice, indicador)
    i += 1


# Prepara layout do corpo do dashboard onde serão exibidos gráficos e tabelas
col = st.columns((6, 2), gap='medium')

# Exibe dos gráficos de pizza
with col[0]:
    
    col1, col2, col3 = st.columns([1,1,1])
    # GRÁFICOS
    renda_list = df_violencia_renda.loc[:, 'RendaTotalFamilia']
    fig = px.pie(df_violencia_renda, values='SofreuViolencia', names=renda_list, title="Violência x Renda Familiar", color_discrete_sequence=px.colors.sequential.Agsunset_r, hole=0)
    col1.plotly_chart(fig, use_container_width=True)

    escolaridade_list = df_violencia_escolaridade.loc[:, 'Escolaridade']
    fig = px.pie(df_violencia_escolaridade, values='SofreuViolencia', names=escolaridade_list, title="Violência x Escolaridade", color_discrete_sequence=px.colors.sequential.Agsunset_r, hole=0)
    col2.plotly_chart(fig, use_container_width=True)

    ocupacao_list = df_violencia_ocupacao.loc[:, 'Ocupacao']
    fig = px.pie(df_violencia_ocupacao, values='SofreuViolencia', names=ocupacao_list, title="Violência x Ocupação", color_discrete_sequence=px.colors.sequential.Agsunset_r, hole=.4)
    col3.plotly_chart(fig, use_container_width=True)

# Exibe a tabela com conforme a categoria
with col[1]:
    if top_selector == 'Estado':
        st.markdown('#### Violência por Estados')

        st.dataframe(df_violencia_estado, 
                    column_order=("Estado", "SofreuViolencia"),
                    hide_index=True,
                    width=420,
                    column_config={
                        "Estado": st.column_config.TextColumn(
                            "Estado", width=120,
                        ),
                        "SofreuViolencia": st.column_config.ProgressColumn(
                            "Sofreu Violencia",
                            format="%f",
                            min_value=0,
                            max_value=max(df_violencia_estado.SofreuViolencia),
                        )}
                    )
    elif top_selector == 'Agressor':
        st.markdown('#### Tipo de Agressor')

        st.dataframe(df_violencia_agressor,
                    column_order=("AgressorViolencia", "SofreuViolencia"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "AgressorViolencia": st.column_config.TextColumn(
                            "Agressor", width=125,
                        ),
                        "SofreuViolencia": st.column_config.ProgressColumn(
                            "Sofreu Violencia",
                            format="%f",
                            min_value=0,
                            max_value=max(df_violencia_estado.SofreuViolencia),
                        )}
                    )
    elif top_selector == 'Estado Civil':
        st.markdown('#### Violência por Estado Civil')

        st.dataframe(df_violencia_estadocivil,
                    column_order=("EstadoCivil", "SofreuViolencia"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "EstadoCivil": st.column_config.TextColumn(
                            "Estado Civil", width=125,
                        ),
                        "SofreuViolencia": st.column_config.ProgressColumn(
                            "Sofreu Violencia",
                            format="%f",
                            min_value=0,
                            max_value=max(df_violencia_estado.SofreuViolencia),
                        )}
                    )
    elif top_selector == 'Região':
        st.markdown('#### Violência por Região')

        st.dataframe(df_violencia_regiao,
                    column_order=("Regiao", "SofreuViolencia"),
                    hide_index=True,
                    width=360,
                    column_config={
                        "Regiao": st.column_config.TextColumn(
                            "Região", width=45,
                        ),
                        "SofreuViolencia": st.column_config.ProgressColumn(
                            "Sofreu Violencia",
                            format="%f",
                            min_value=0,
                            max_value=max(df_violencia_estado.SofreuViolencia),
                        )}
                    )
    elif top_selector == 'Etnia':
        st.markdown('#### Violência por Etnia')

        st.dataframe(df_violencia_etnia,
                    column_order=("CorRaca", "SofreuViolencia"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "CorRaca": st.column_config.TextColumn(
                            "Etnia", width=125,
                        ),
                        "SofreuViolencia": st.column_config.ProgressColumn(
                            "Sofreu Violencia",
                            format="%f",
                            min_value=0,
                            max_value=max(df_violencia_estado.SofreuViolencia),
                        )}
                    )

    elif top_selector == 'Religião/Crença':
        st.markdown('#### Violência por Religião/Crença')

        st.dataframe(df_violencia_religiao,
                    column_order=("ReligiaoCrenca", "SofreuViolencia"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "ReligiaoCrenca": st.column_config.TextColumn(
                            "Religião/Crença", width=125,
                        ),
                        "SofreuViolencia": st.column_config.ProgressColumn(
                            "Sofreu Violencia",
                            format="%f",
                            min_value=0,
                            max_value=max(df_violencia_estado.SofreuViolencia),
                        )}
                    )

    elif top_selector == 'Posicionamento Político':
        st.markdown('#### Violência por Posicionamento Político')

        st.dataframe(df_violencia_politica,
                    column_order=("PosicionamentoPolitico", "SofreuViolencia"),
                    hide_index=True,
                    width=None,
                    column_config={
                        "PosicionamentoPolitico": st.column_config.TextColumn(
                            "Posicionamento Político", width=125,
                        ),
                        "SofreuViolencia": st.column_config.ProgressColumn(
                            "Sofreu Violencia",
                            format="%f",
                            min_value=0,
                            max_value=max(df_violencia_estado.SofreuViolencia),
                        )}
                    )
