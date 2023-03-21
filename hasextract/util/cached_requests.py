import logging
from redis import Redis
import requests


redis = Redis(decode_responses=True)

logger = logging.getLogger()


def get(url: str, headers, timeout=None, key=None, data=None, json=None):
    if not key:
        key = url
    page_text = redis.get(key)
    try:
        if not page_text:
            result = requests.get(
                url, headers=headers, timeout=timeout, data=data, json=json
            )
            if result.status_code < 400:
                page_text = result.text
                if page_text is not None:
                    redis.set(key, page_text)
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


def post(url: str, headers, data=None, json=None, timeout=None, key=None):
    if not key:
        key = url
    page_text = redis.get(key)
    try:
        if not page_text:
            result = requests.post(
                url, headers=headers, timeout=timeout, data=data, json=json
            )
            if result.status_code < 400:
                page_text = result.text
                if page_text is not None:
                    redis.set(key, page_text)
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


