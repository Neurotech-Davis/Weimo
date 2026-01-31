# To restart, you need to make this an executable. chmod +x ./restart.sh
docker-compose down          # Takes down the running containers (allowing refreshing). -v to delete volumes
export DOCKER_BUILDKIT=0
docker-compose up --build -d # Runs containers, initializing them. -d is detach
sleep 1                      # if it runs instantly it can't tell if it will crash or not
docker ps -a