
Set up the backend server on the host machine:

```bash
./restart.sh
```

Setting up Ngrok
* "If it works on localhost, it will work on Ngrok"
* Ngrok is like a middle-man that handles tunneling. Here's the flow for traffic:
  * My laptop's localhost -> Ngrok -> My apartment's server running on locahost