# USER GUIDE

1. Follow the instructions 
[here](https://developers.google.com/gmail/api/quickstart/python)
to enable the Gmail API and generate OAuth Key.
2. Add the credentials of Gmail API OAuth key in `conf/credentials.json`
3. Use the `requirements.txt` to create a pip virtual environment:
    ```
    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
4. Run `main.py`
5. Either use the model trained on my mails. you have to create those labels by using 
option 9 of menu \
or you should label most of your mails and train the model on 
your mails by using option 7 of menu
