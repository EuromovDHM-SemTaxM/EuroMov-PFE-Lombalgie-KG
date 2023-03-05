from redis import Redis
import requests


redis = Redis(decode_responses=True)



def get(url: str, headers, timeout= None):
    page_text = redis.get(url)
    try:
        if not page_text:
            result = requests.get(url, headers=headers, timeout=timeout)
            if result.status_code < 400:
                page_text = result.text
                if page_text is not None:
                    redis.set(url, page_text)
            else:
                print(result, result.text)
                return None
    except requests.exceptions.ReadTimeout:
        page_text = None
    except requests.exceptions.MissingSchema:
        page_text = None
    except requests.exceptions.ConnectTimeout:
        page_text = None
    return page_text

def post(url: str, headers, data, timeout= None):
    page_text = redis.get(url)
    try:
        if not page_text:
            result = requests.post(url, headers=headers, timeout=timeout, data=data)
            if result.status_code < 400:
                page_text = result.text
                if page_text is not None:
                    redis.set(url, page_text)
            else:
                print(result, result.text)
                return None
    except requests.exceptions.ReadTimeout:
        page_text = None
    except requests.exceptions.MissingSchema:
        page_text = None
    except requests.exceptions.ConnectTimeout:
        page_text = None
    return page_text