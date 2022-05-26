# scoring-bank-p7-dashboard
Dashboard du projet P7 Scoring Bank
* https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app

**https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app
**https://towardsdatascience.com/a-quick-tutorial-on-how-to-deploy-your-streamlit-app-to-heroku-874e1250dadd
#### Step 1. Create a new app on Streamlit/Heroku "scoring-bank-p7-dashboard"

#### Step 2. Initiate Project Folder on Local PC
* Test before commit
* [any folder pathname] > cd [project folder pathname]
* [project folder pathname] > python -m venv venv
* [project folder pathname] > CALL venv/Scripts/activate.bat
* [project folder pathname] > python -m pip install --upgrade pip
* [project folder pathname] > pip install numpy pandas streamlit shap matplotlib seaborn
* [project folder pathname] > pip freeze > requirements.txt
** Change the following line in requirements.txt : pywin32==304;platform_system == "Windows"

#### Step 3. Initiate GIT in the project folder
* [project folder pathname] > git init
* [project folder pathname] > git add .
* [project folder pathname] > git commit -m "first commit"

** [project folder pathname] > heroku login
** [project folder pathname] > heroku create scoring-bank-p7-dashboard
** [project folder pathname] > git push heroku main

** [project folder pathname] > heroku login
** [project folder pathname] > heroku create scoring-bank-p7-dashboard
** [project folder pathname] > git push heroku master

#### Step 4. Testing application
* Heroku dashboard https://dashboard.heroku.com/apps/scoring-bank-p7-dashboard
* accessing app https://scoring-bank-p7-dashboard.herokuapp.com/
* checking logs https://dashboard.heroku.com/apps/scoring-bank-p7-dashboard/logs
