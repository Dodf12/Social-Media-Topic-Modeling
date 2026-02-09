API_BASE="https://api.substackapi.dev/"
API_KEY="sk_live_c940059c97c54ce39bc32d4dda2d2e67"
Substack_Publication="https://dodf12.substack.com/"

curl https://api.substackapi.dev/api_key/validate \
-H "X-API-Key: ${API_KEY}"

curl "https://api.substackapi.dev/posts/latest?publication_url=dodf12.substack.com" \
-H "X-API-Key: ${API_KEY}" > output.txt