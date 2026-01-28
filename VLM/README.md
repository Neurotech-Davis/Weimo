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

5. Set up Ngrok for tunneling
```bash
# On remote machine

```

