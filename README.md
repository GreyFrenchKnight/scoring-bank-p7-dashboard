# scoring-bank-p7-dashboard
Dashboard du projet P7 Scoring Bank
* https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app
#### Web application deployed at: https://share.streamlit.io/greyfrenchknight/scoring-bank-p7-dashboard/main/P7_dashboard.py

#### Step 1. Create a new app on Streamlit "scoring-bank-p7-dashboard"

#### Step 2. Initiate Project Folder on Local PC
* Test before commit
* [any folder pathname] > cd [project folder pathname]
* [project folder pathname] > python -m venv venv
* [project folder pathname] > CALL venv/Scripts/activate.bat
* [project folder pathname] > python -m pip install --upgrade pip
* [project folder pathname] > pip install numpy pandas shap matplotlib seaborn streamlit
* [project folder pathname] > pip freeze > requirements.txt

* Change the following line in requirements.txt : pywin32==304;platform_system == "Windows"

#### Step 3. Initiate GIT in the project folder
* [project folder pathname] > git init
* [project folder pathname] > git add .
* [project folder pathname] > git commit -m "first commit"

#### Step 4. Deploy app
To deploy an app, click "New app" from the upper right corner of your workspace, then fill in your repo, branch, and file path, and click "Deploy". As a shortcut, you can also click "Paste GitHub URL".
![deployment_streamlit](https://github.com/GreyFrenchKnight/scoring-bank-p7-dashboard/blob/f0df22a5040b12fc0506d47cf5d2a262e201c95d/streamlit_deployment.PNG)

#### Step 5. Testing application
* accessing app https://share.streamlit.io/greyfrenchknight/scoring-bank-p7-dashboard/main/P7_dashboard.py
