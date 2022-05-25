#import joblib
import streamlit as st

import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import shap
#import matplotlib.pyplot as plt
#import plotly.graph_objects as go
#from sklearn import model_selection
#from sklearn.neighbors import NearestNeighbors
#import seaborn as sns
#import lightgbm
#from matplotlib.image import imread

import requests
import io
import json

from PIL import Image

# https://realpython.com/fastapi-python-web-apis/
# launch Dashboard in local
# cd C:\Users\disch\Documents\GitHub\scoring-bank-p7-dashboard
# streamlit run P7_dashboard

# Basic URL
# http://localhost:8501/

st.set_page_config(layout='wide')

main_title = 'Dashboard Pret à dépenser'
first_page = 'Tous les clients'
second_page = 'Score Client'
third_page = 'Comparaison Clients'

PAGES = [
    first_page,
    second_page,
    third_page
]

# Initialization
if 'selected_page' not in st.session_state:
    st.session_state['selected_page'] = first_page
if 'selected_page_index' not in st.session_state:
    st.session_state['selected_page_index'] = 0        
    
"st.session_state object", st.session_state

def load_data():
    '''Fonction chargeant et calculant les données nécessaires au dashboard.
    Ne prend pas de paramètres en entrée
    '''
    # Chargement des noms de colonnes pour la prédiction du model
    r = requests.get('http://127.0.0.1:8000/features_for_model_prediction')
    features_for_model_prediction = r.json()
    print("\nFeatures for model prediction:", features_for_model_prediction)
    
    # Chargement des noms de colonnes pour l'affichage du tableau
    r = requests.get('http://127.0.0.1:8000/features_for_dashboard_table')
    features_for_dashboard_table = r.json()
    print("\nFeatures for dashboard main table:", features_for_dashboard_table)
    
    # Chargement des données de test
    r = requests.get('http://127.0.0.1:8000/clients')
    db_test = pd.read_json(str(r.json()).replace("'", '"'))
    db_test = db_test.reset_index(drop=True)
    
    # Initialization
    if 'selected_client' not in st.session_state:
        st.session_state['selected_client'] = db_test['SK_ID_CURR'][0]
    
    # Chargement des SHAP values
    r = requests.get('http://127.0.0.1:8000/shap_shap_values')
    inmemoryfile = io.BytesIO(r.content)
    shap_values = np.genfromtxt(inmemoryfile, dtype=float, delimiter=" ")
    print("\nSHAP SHAP value:", shap_values)
    
    r = requests.get('http://127.0.0.1:8000/shap_expected_value')
    exp_value = r.json()
    print("\nSHAP expected value:", exp_value)
    
    # Directly reading images from URLs is deprecated since 3.4 and will no 
    # longer be supported two minor releases later. Please open the URL for
    # reading and pass the result to Pillow, e.g. with
    # ``np.array(PIL.Image.open(urllib.request.urlopen(url)))``.
    logo = Image.open("data/logo.png")  
    
    return db_test, logo, features_for_model_prediction, features_for_dashboard_table, shap_values, exp_value


def get_client(db_test):
    """Sélection d'un client via une selectbox"""
    client = st.sidebar.selectbox('Client', db_test['SK_ID_CURR'], key="selected_client")
    idx_client = db_test.index[db_test['SK_ID_CURR'] == client][0]
    return client, idx_client


def infos_client(db_test, client, idx_client):
    """Affichage des infos du client sélectionné dans la barre latérale"""
    st.sidebar.markdown("**ID client:** " + str(client))
    st.sidebar.markdown("**Sexe:** " + db_test.loc[idx_client,'CODE_GENDER'])
    st.sidebar.markdown("**Statut familial:** " + db_test.loc[idx_client,'NAME_FAMILY_STATUS'])
    st.sidebar.markdown("**Enfants:** " + str(db_test.loc[idx_client,'CNT_CHILDREN']))
    st.sidebar.markdown("**Age:** " + str(db_test.loc[idx_client,'YEARS_BIRTH']))	
    st.sidebar.markdown("**Statut pro.:** " + db_test.loc[idx_client,'NAME_INCOME_TYPE'])
    st.sidebar.markdown("**Niveau d'études:** " + db_test.loc[idx_client,'NAME_EDUCATION_TYPE'])
    

def tab_client(db_test, features_for_dashboard_table):

    '''Fonction pour afficher le tableau du portefeuille client avec un système de 6 champs de filtres
    permettant une recherche plus précise.
    Le paramètre est le dataframe des clients
    '''
    st.title(main_title)
    st.subheader(first_page)
    
    row0_1,row0_spacer2,row0_2,row0_spacer3,row0_3,row0_spacer4,row0_4,row0_spacer5,row0_5,row0_spacer6 = st.columns([1,.1,1,.1,1,.1,1,.1,1,.1])

	#Définition des filtres via selectbox    
    sex = row0_1.selectbox("Genre",['All']+db_test['CODE_GENDER'].unique().tolist())
    age = row0_2.selectbox("Age",['All']+(np.sort(db_test['YEARS_BIRTH'].unique()).astype(str).tolist()))
    fam = row0_3.selectbox("Statut familial",['All']+db_test['NAME_FAMILY_STATUS'].unique().tolist())
    child = row0_4.selectbox("Enfants",['All']+(np.sort(db_test['CNT_CHILDREN'].unique()).astype(str).tolist()))
    stud = row0_5.selectbox("Niveau d'études",['All']+db_test['NAME_EDUCATION_TYPE'].unique().tolist())
    car = row0_1.selectbox("Véhiculé",['All']+db_test['FLAG_OWN_CAR'].unique().tolist())
    realty = row0_2.selectbox("Propriétaire",['All']+db_test['FLAG_OWN_REALTY'].unique().tolist())
    housing = row0_3.selectbox("Type de logement",['All']+db_test['NAME_HOUSING_TYPE'].unique().tolist())
    pro = row0_4.selectbox("Source de revenu",['All']+db_test['NAME_INCOME_TYPE'].unique().tolist())
    
    row0_1,row0_spacer2,row0_2,row0_spacer3,row0_3,row0_spacer4 = st.columns([2,.1,2,.1,2,.1])
    total_income = row0_1.slider(label="Revenus totaux", min_value=db_test['AMT_INCOME_TOTAL'].min(), max_value=db_test['AMT_INCOME_TOTAL'].max(), value=(float(db_test['AMT_INCOME_TOTAL'].min()), float(db_test['AMT_INCOME_TOTAL'].max())), step=1000.)
    credit = row0_2.slider(label="Crédit", min_value=db_test['AMT_CREDIT'].min(), max_value=db_test['AMT_CREDIT'].max(), value=(float(db_test['AMT_CREDIT'].min()), float(db_test['AMT_CREDIT'].max())), step=1000.)
    annuity = row0_3.slider(label="Annuité", min_value=db_test['AMT_ANNUITY'].min(), max_value=db_test['AMT_ANNUITY'].max(), value=(float(db_test['AMT_ANNUITY'].min()), float(db_test['AMT_ANNUITY'].max())), step=1000.)
    
    #Affichage du dataframe selon les filtres définis
    db_display = db_test[features_for_dashboard_table].copy()    
    db_display['YEARS_BIRTH'] = db_display['YEARS_BIRTH'].astype(str)
    db_display['CNT_CHILDREN'] = db_display['CNT_CHILDREN'].astype(str)
    db_display['AMT_INCOME_TOTAL'] = db_display['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
    db_display['AMT_CREDIT'] = db_display['AMT_CREDIT'].apply(lambda x: int(x))
    db_display['AMT_ANNUITY'] = db_display['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))
    db_display['AMT_INCOME_TOTAL'] = db_display['AMT_INCOME_TOTAL'].apply(lambda x: x if pd.isna(x) else int(x))
    db_display['AMT_CREDIT'] = db_display['AMT_CREDIT'].apply(lambda x: x if pd.isna(x) else int(x))
    db_display['AMT_ANNUITY'] = db_display['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))
    
    # filtering
    db_display = filter(db_display,'CODE_GENDER', sex)
    db_display = filter(db_display,'YEARS_BIRTH', age)
    db_display = filter(db_display,'NAME_FAMILY_STATUS', fam)
    db_display = filter(db_display,'CNT_CHILDREN', child)
    db_display = filter(db_display,'NAME_EDUCATION_TYPE', stud)
    db_display = filter(db_display,'FLAG_OWN_CAR', car)
    db_display = filter(db_display,'FLAG_OWN_REALTY', realty)
    db_display = filter(db_display,'NAME_HOUSING_TYPE', housing)
    db_display = filter(db_display,'NAME_INCOME_TYPE', pro)
    db_display = filter(db_display,'AMT_INCOME_TOTAL', total_income)
    db_display = filter(db_display,'AMT_CREDIT', credit)
    db_display = filter(db_display,'AMT_ANNUITY', annuity)
    
    st.dataframe(db_display)
    st.markdown("**Total clients correspondants:** " + str(len(db_display)))
   
    # allows user to filter then select one row
    selected_indices = st.multiselect('Select row:', db_test['SK_ID_CURR'])
    
    # change to second page with currently selected client row
    if len(selected_indices) == 1:
        print(selected_indices[0])
        if st.button(second_page):
            st.session_state['selected_page_index'] = 1   
            st.session_state['selected_page'] = second_page
            st.session_state['selected_client'] = selected_indices[0]
            st.experimental_rerun()    
    
def filter(df, col, value):
    '''Fonction pour filtrer le dataframe selon la colonne et la valeur définies'''
    if value!='All':
        # 1 value or tuple value
        if type(value) is tuple:
            # tuple for numeric, use of min/max
            db_filtered=df.loc[(df[col] >= value[0]) & (df[col] <= value[1])]
        else:
            db_filtered=df.loc[df[col]==value]
    else:
        db_filtered=df
    return db_filtered


def comparaison(db_test, idx_client):
    """Fonction principale de l'onglet 'Comparaison clientèle' """
    st.title(main_title)
    st.subheader(third_page)
    
    print("\nTODO")
    
    print("\n")
    
    
def score_viz(db_test, idx_client, features_for_model_prediction, shap_values, exp_value):
    """Fonction principale de l'onglet 'Score visualisation' """
    st.title(main_title)
    st.subheader(second_page)
    
    # Prediction and SHAP values
    
    # preparing POST json
    client_features_for_model_prediction = db_test.loc[idx_client, features_for_model_prediction].to_dict()
    jsoned = json.dumps(client_features_for_model_prediction)
    
    # POST
    r = requests.post(url='http://127.0.0.1:8000/loan_repayment', data=jsoned)
    
    # Response
    response = json.loads(r.json())
    y_pred = float(response["y_pred"]) 
    decision = response["decision"]
    features_client = pd.read_json(str(response["features"]).replace("'", '"'))
    features_client = features_client.reset_index(drop=True)
    feature_names = response["feature_names"]    
    
    st.caption("**Score :** \t{}".format(y_pred))
    st.caption("**Crédit :** \t{}".format(decision))
    
    # SHAP plot
    st.subheader("Graphique SHAP")
    st_shap(shap.force_plot(exp_value, shap_values[idx_client], features=features_client, feature_names=feature_names, figsize=(12,5)))
    
    st.caption("**Explications du graphique SHAP concernant la modélisation**")
    col1_red_refuse, col2_blue_accept = st.columns(2)
    col1_red_refuse.caption("<font color=‘#ff2624’>En rouge, les variables qui ont un impact positif (contribuent à ce que la prédiction soit plus élevée que la valeur de base), donc qui tendent vers un non remboursement du crédit (TARGET = 1)</font>", True)
    col2_blue_accept.caption("<font color=blue>En bleu, les variables qui ont un impact négatif (contribuent à ce que la prédiction soit plus basse que la valeur de base), donc qui tendent vers un remboursement du crédit (TARGET = 0)</font>", True)
    
    
def st_shap(plot, height=None):
	"""Fonction permettant l'affichage de graphique shap values"""
	shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
	components.html(shap_html, height=height)


def page_changed():
    st.session_state.selected_page_index = PAGES.index(st.session_state.selected_page)
    print('current page index', st.session_state.selected_page_index)

 
def main():
    """Fonction principale permettant l'affichage de la fenêtre latérale avec les 3 onglets.
    """
    db_test, logo, features_for_model_prediction, features_for_dashboard_table, shap_values, exp_value = load_data()
    st.sidebar.image(logo)
    st.sidebar.write('')
    st.sidebar.write('')

    st.sidebar.title('Menu')
    st.sidebar.radio(label="", options=PAGES, index=st.session_state.selected_page_index, key="selected_page", on_change=page_changed)
        
    if st.session_state.selected_page == first_page:
        tab_client(db_test, features_for_dashboard_table)
    if st.session_state.selected_page == second_page:
        client,idx_client=get_client(db_test)
        infos_client(db_test, client, idx_client)
        score_viz(db_test, idx_client, features_for_model_prediction, shap_values, exp_value)
    if st.session_state.selected_page == third_page:
        client,idx_client=get_client(db_test)
        infos_client(db_test, client, idx_client)
        comparaison(db_test, idx_client)

if __name__ == '__main__':
    main()
