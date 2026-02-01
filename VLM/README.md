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
```
* Go into browser and paste this in:
```
http://0.0.0.0:8000/ping
```
* Make sure it returns "pong"

5. Set up Ngrok for tunneling
    - First time: Follow the instructions [here](https://ngrok.com/download/linux?tab=install)
        - Make sure you use the correct port (I used 8000 for backend)
        - Once it's set up, copy and paste the https request for and call the API.
    - Next time: 
```bash
# On remote machine
ngrok config add-authtoken {auth_token}
ngrok http 8000
```