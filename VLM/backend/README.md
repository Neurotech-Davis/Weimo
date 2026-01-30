
## 1) Set up the backend server on the host machine

```bash
# On remote machine
cd backend/
./restart.sh
```
* Go into browser and paste this in:
```
http://0.0.0.0:8000/ping
```
* Make sure it returns "pong"

NOTE: Might have to remove "AutoModelForVision2Seq"
TODO: There might be a better way to organize this backend project

## 2) Setting up Ngrok
* "If it works on localhost, it will work on Ngrok"
* Ngrok is like a middle-man that handles tunneling. Here's the flow for traffic:
  * My laptop's localhost -> Ngrok -> My apartment's server running on locahost
