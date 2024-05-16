echo off
rem Author: God Bennett
rem Note: If you can't run this batch, you can simply 
rem (1) open cmd
rem (2) navigate to the ai location eg "cd C:\Users\god.bennett\Downloads\Ai_CreditCardEtc_FraudDetection-main"
rem (3) set path eg "path=C:\Users\god.bennett\Downloads\python-3.6.3-embed-amd64"
rem (4) run ui core eg "python UI_CORE.py" then wait till ui loads up

rem set path to python 3.6.3
rem %AppData% + \Local\Programs\Python\Python36
path=C:\Users\god.bennett\Downloads\python-3.6.3-embed-amd64

rem load ai model for inference\aka answer retrieval
echo loading artificial intelligence model...
python UI_CORE.py



pause
