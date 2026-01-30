import requests

# Hit this URL
url = "http://localhost:8000/api/v1/vlm/analyze/"

# Format prompt
files = {"image": open("../images/example.png", "rb")}
data = { "system_prompt": "", "message_prompt": "" }

# Get
response = requests.post(url, data=data, files=files["image"])
print(response.json())
