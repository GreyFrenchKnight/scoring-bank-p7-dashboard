import joblib
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn import model_selection
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
import lightgbm
from matplotlib.image import imread

import requests

# https://realpython.com/fastapi-python-web-apis/
# launch Dashboard in local
# cd C:\Users\disch\Documents\GitHub\scoring-bank-p7-dashboard
# streamlit run P7_dashboard

# Basic URL
# http://localhost:8501/

st.set_page_config(layout="wide")

main_title = 'Dashboard Pret à dépenser'
first_page = 'Tous les clients'
second_page = "Score Client"
third_page = 'Comparaison Clients'

def load_data():
    '''Fonction chargeant et calculant les données nécessaires au dashboard.
    Ne prend pas de paramètres en entrée
    '''

    # Chargement du modèle pré-entrainé	
    # pickle_in = dvc.api.read('Data/lgbm_model.pkl',
    # repo='https://github.com/StevPav/OCR_Data_Scientist_P7.git',
    # mode='rb')
    # lgbm=pickle.loads(pickle_in)

    #Chargement des données de test
    r = requests.get('http://127.0.0.1:8000/clients')
    print(r.status_code)
    print(r.json())
    
    db_test = pd.read_json(str(r.json()).replace("'", '"'))
    db_test = db_test.reset_index(drop=True)
    # df_test=pd.read_csv('https://github.com/StevPav/OCR_Data_Scientist_P7/blob/70ed8a21e2d52f1d7c927f0ea5d496e5c77de617/Data/df_test.csv?raw=true')
    logo = imread("https://github.com/GreyFrenchKnight/scoring-bank-p7-dashboard/blob/feeebc2dbc42a932b20dba0202b03f45f6de500a/data/logo.png?raw=true")

    #Calcul des SHAP values
    # explainer = shap.TreeExplainer(lgbm)
    # shap_values = explainer.shap_values(df_test)[1]
    # exp_value=explainer.expected_value[1]
    return db_test, logo

def get_client(db_test):
    """Sélection d'un client via une selectbox"""
    client=st.sidebar.selectbox('Client',db_test['SK_ID_CURR'])
    idx_client=db_test.index[db_test['SK_ID_CURR']==client][0]
    return client,idx_client

def infos_client(db_test,client,idx_client):
    """Affichage des infos du client sélectionné dans la barre latérale"""
    st.sidebar.markdown("**ID client: **"+str(client))
    st.sidebar.markdown("**Sexe: **"+db_test.loc[idx_client,'CODE_GENDER'])
    st.sidebar.markdown("**Statut familial: **"+db_test.loc[idx_client,'NAME_FAMILY_STATUS'])
    st.sidebar.markdown("**Enfants: **"+str(db_test.loc[idx_client,'CNT_CHILDREN']))
    st.sidebar.markdown("**Age: **"+str(db_test.loc[idx_client,'YEARS_BIRTH']))	
    st.sidebar.markdown("**Statut pro.: **"+db_test.loc[idx_client,'NAME_INCOME_TYPE'])
    st.sidebar.markdown("**Niveau d'études: **"+db_test.loc[idx_client,'NAME_EDUCATION_TYPE'])

def tab_client(db_test):

    '''Fonction pour afficher le tableau du portefeuille client avec un système de 6 champs de filtres
    permettant une recherche plus précise.
    Le paramètre est le dataframe des clients
    '''
    st.title(main_title)
    st.subheader(first_page)
    
    row0_1,row0_spacer2,row0_2,row0_spacer3,row0_3,row0_spacer4,row_spacer5 = st.columns([1,.1,1,.1,1,.1,4])

	#Définition des filtres via selectbox
    sex=row0_1.selectbox("Sexe",['All']+db_test['CODE_GENDER'].unique().tolist())
    age=row0_1.selectbox("Age",['All']+(np.sort(db_test['YEARS_BIRTH'].unique()).astype(str).tolist()))
    fam=row0_2.selectbox("Statut familial",['All']+db_test['NAME_FAMILY_STATUS'].unique().tolist())
    child=row0_2.selectbox("Enfants",['All']+(np.sort(db_test['CNT_CHILDREN'].unique()).astype(str).tolist()))
    pro=row0_3.selectbox("Statut pro.",['All']+db_test['NAME_INCOME_TYPE'].unique().tolist())
    stud=row0_3.selectbox("Niveau d'études",['All']+db_test['NAME_EDUCATION_TYPE'].unique().tolist())
    
    #Affichage du dataframe selon les filtres définis
    db_display=db_test[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
    'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
    db_display['YEARS_BIRTH']=db_display['YEARS_BIRTH'].astype(str)
    db_display['CNT_CHILDREN']=db_display['CNT_CHILDREN'].astype(str)
    db_display['AMT_INCOME_TOTAL']=db_test['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
    db_display['AMT_CREDIT']=db_test['AMT_CREDIT'].apply(lambda x: int(x))
    db_display['AMT_ANNUITY']=db_test['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))
    
    db_display=filter(db_display,'CODE_GENDER',sex)
    db_display=filter(db_display,'YEARS_BIRTH',age)
    db_display=filter(db_display,'NAME_FAMILY_STATUS',fam)
    db_display=filter(db_display,'CNT_CHILDREN',child)
    db_display=filter(db_display,'NAME_INCOME_TYPE',pro)
    db_display=filter(db_display,'NAME_EDUCATION_TYPE',stud)
    
    st.dataframe(db_display)
    st.markdown("**Total clients correspondants: **"+str(len(db_display)))
    
    
def filter(df,col,value):
	'''Fonction pour filtrer le dataframe selon la colonne et la valeur définies'''
	if value!='All':
		db_filtered=df.loc[df[col]==value]
	else:
		db_filtered=df
	return db_filtered

def comparaison(db_test,idx_client):
    """Fonction principale de l'onglet 'Comparaison clientèle' """
    st.title(main_title)
    st.subheader(second_page)
    
def score_viz(client):
    """Fonction principale de l'onglet 'Score visualisation' """
    st.title(main_title)
    st.subheader(third_page)
    
def main():
    """Fonction principale permettant l'affichage de la fenêtre latérale avec les 3 onglets.
    """
    db_test, logo=load_data()
    st.sidebar.image(logo)
    PAGES = [
        first_page,
        second_page,
        third_page
    ]

    st.sidebar.write('')
    st.sidebar.write('')

    st.sidebar.title('Pages')
    selection = st.sidebar.radio("Go to", PAGES)

    if selection==first_page:
        tab_client(db_test)
    if selection==second_page:
        client,idx_client=get_client(db_test)
        infos_client(db_test,client,idx_client)
        score_viz(idx_client)
    if selection==third_page:
        client,idx_client=get_client(db_test)
        infos_client(db_test,client,idx_client)
        comparaison(db_test,idx_client)

if __name__ == '__main__':
    main()
