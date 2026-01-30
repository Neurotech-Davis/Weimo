# VLM

Setting up SSH (assuming linux environment):

1. Start a remote SSH server on the GPU remote desktop
```bash
# On remote machine
sudo systemctl start ssh
sudo systemctl enable ssh
sudo systemctl status ssh
sudo systemctl status sshd
```

2. Install Tailscale on both devices
```bash
# On both machines
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

3. Find Tailscale IP & ssh
```bash
# On remote machine
tailscale ip
```

```bash
# On edge device
ssh chengyi@tailscale_ip
```

4. Spin up the backend
```bash
# On remote machine
cd backend/
./restart.sh
uvicorn src.app.main:app --proxy-headers --host 0.0.0.0 --port 8000
```
* Go into browser and paste this in:
```
http://0.0.0.0:800/ping
```
* Make sure it returns "pong"

5. Set up Ngrok for tunneling
```bash
# On remote machine
sudo snap install ngrok
ngrok config add-authtoken <token> # Grab token by signing up for an account
ngrok http 80 # This is where we held the backend
```

