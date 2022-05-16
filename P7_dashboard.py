import pandas as pd
import streamlit as st


# https://realpython.com/fastapi-python-web-apis/
# launch API in local
# cd C:\Users\disch\Documents\OpenClassrooms\Workspace\20220411-Projet_7_Implementez_un_modele_de_scoring\Projet_7
# streamlit run P7_dashboard

# Basic URL
# http://localhost:8501/

st.set_page_config(layout="wide")


def load_data():
	'''Fonction chargeant et calculant les données nécessaires au dashboard.
	Ne prend pas de paramètres en entrée
	'''
    
    #
    # Deserializing
    #

    #ohe = joblib.load('bin/ohe.joblib')
    #scaler = joblib.load('bin/std_scaler.joblib')
    #model = joblib.load('bin/model.joblib')
    
    #
    # Building Model
    #
    
    #data_model_dict = joblib.load('bin/data_dict.joblib')
    #ClientModel = create_model("ClientModel", **data_model_dict)
    
	# Chargement du modèle pré-entrainé	
	pickle_in = dvc.api.read('bin/data_dict.joblib',
        repo='https://github.com/GreyFrenchKnight/scoring-bank-p7.git',
        mode='rb')
	lgbm=pickle.loads(pickle_in)

	#Chargement des données de test
	db_test=pd.read_csv('https://github.com/StevPav/OCR_Data_Scientist_P7/blob/70ed8a21e2d52f1d7c927f0ea5d496e5c77de617/Data/df_app.csv?raw=true')
	db_test['YEARS_BIRTH']=(db_test['DAYS_BIRTH']/-365).apply(lambda x: int(x))
	db_test=db_test.reset_index(drop=True)
	df_test=pd.read_csv('https://github.com/StevPav/OCR_Data_Scientist_P7/blob/70ed8a21e2d52f1d7c927f0ea5d496e5c77de617/Data/df_test.csv?raw=true')
	logo=imread("https://github.com/StevPav/OCR_Data_Scientist_P7/blob/70ed8a21e2d52f1d7c927f0ea5d496e5c77de617/Data/logo.png?raw=true")

	#Calcul des SHAP values
	explainer = shap.TreeExplainer(lgbm)
	shap_values = explainer.shap_values(df_test)[1]
	exp_value=explainer.expected_value[1]
	return db_test,df_test,shap_values,lgbm,exp_value,logo

#
# Data
#

data = pd.read_csv('data/application_train.csv')
data['YEARS_BIRTH'] = data['DAYS_BIRTH'] / -365

#
# Dashboard
#

st.title('Dashboard Pret à dépenser')
st.subheader('Tableau clientèle')
row0_1,row0_spacer2,row0_2,row0_spacer3,row0_3,row0_spacer4,row_spacer5 = st.columns([1,.1,1,.1,1,.1,4])

#Définition des filtres via selectbox
# sex=row0_1.selectbox("Sexe",['All']+db_test['CODE_GENDER'].unique().tolist())
# age=row0_1.selectbox("Age",['All']+(np.sort(db_test['YEARS_BIRTH'].unique()).astype(str).tolist()))
# fam=row0_2.selectbox("Statut familial",['All']+db_test['NAME_FAMILY_STATUS'].unique().tolist())
# child=row0_2.selectbox("Enfants",['All']+(np.sort(db_test['CNT_CHILDREN'].unique()).astype(str).tolist()))
# pro=row0_3.selectbox("Statut pro.",['All']+db_test['NAME_INCOME_TYPE'].unique().tolist())
# stud=row0_3.selectbox("Niveau d'études",['All']+db_test['NAME_EDUCATION_TYPE'].unique().tolist())

# #Affichage du dataframe selon les filtres définis
db_display=data[['SK_ID_CURR','CODE_GENDER','YEARS_BIRTH','NAME_FAMILY_STATUS','CNT_CHILDREN',
'NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
'NAME_INCOME_TYPE','AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY']]
db_display['YEARS_BIRTH']=db_display['YEARS_BIRTH'].astype(str)
db_display['CNT_CHILDREN']=db_display['CNT_CHILDREN'].astype(str)
db_display['AMT_INCOME_TOTAL']=db_display['AMT_INCOME_TOTAL'].apply(lambda x: int(x))
db_display['AMT_CREDIT']=db_display['AMT_CREDIT'].apply(lambda x: int(x))
db_display['AMT_ANNUITY']=db_display['AMT_ANNUITY'].apply(lambda x: x if pd.isna(x) else int(x))

# db_display=filter(db_display,'CODE_GENDER',sex)
# db_display=filter(db_display,'YEARS_BIRTH',age)
# db_display=filter(db_display,'NAME_FAMILY_STATUS',fam)
# db_display=filter(db_display,'CNT_CHILDREN',child)
# db_display=filter(db_display,'NAME_INCOME_TYPE',pro)
# db_display=filter(db_display,'NAME_EDUCATION_TYPE',stud)

st.dataframe(db_display)
st.markdown("**Total clients correspondants: **"+str(len(db_display)))
