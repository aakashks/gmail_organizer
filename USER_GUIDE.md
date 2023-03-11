# USER GUIDE

1. Follow the instructions 
[here](https://developers.google.com/gmail/api/quickstart/python)
to enable the Gmail API and generate OAuth Key.
2. Add the credentials of Gmail API OAuth key in `conf/credentials.json`
3. Use the resulting `requirements.txt` to create a pip virtual environment:
    ```
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
4. Run `main.py`