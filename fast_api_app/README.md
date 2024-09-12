## Get fast api app running
- Cmd in location of Dockerfile: docker compose up --build

## Send a statement 
curl -X 'POST' \
  'http://0.0.0.0:8001/statements/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "statement": "Example statment: I am scared of the future!"
}'

## See example of official docker docs https://docs.docker.com/language/python/develop/