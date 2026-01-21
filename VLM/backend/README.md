```bash
docker-compose down -v                             # Takes down the running containers (allowing refreshing)
docker-compose up --build -d                       # Runs containers, initializing them
docker-compose logs {name of container (api)} # Debugging purposes
```