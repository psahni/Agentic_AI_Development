# Load the python env

## Mac / Linux
source .venv/bin/activate

## Windows Command Prompt
.venv\Scripts\activate.bat

## Windows PowerShell
.venv\Scripts\Activate.ps1


# Get the Gemini api Key
https://aistudio.google.com/


# Enable Gmail API & get credentials

* Go to console.cloud.google.com
* Create a new project (or select existing)
* Search "Gmail API" → click Enable
* Go to "Credentials" → "Create Credentials" → "OAuth Client ID"
* Application type → Desktop App → click Create
* Download the JSON file → rename it to credentials.json
* Place credentials.json in your project folder

# Install packages
$ pip install google-generativeai google-api-python-client google-auth-httplib2 google-auth-oauthlib
