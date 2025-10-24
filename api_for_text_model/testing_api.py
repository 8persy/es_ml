import requests

try:
    r = requests.post(
        "http://localhost:8000/predict",
        json={"texts": ["Очень понравился фильм!", "Не подошло, скучно"]}
    )
except requests.exceptions.RequestException as e:
    print(f"Error with request: {e}")
    exit(1)
    
print(r.status_code)
print(r.json())