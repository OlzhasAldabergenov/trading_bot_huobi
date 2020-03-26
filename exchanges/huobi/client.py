
import base64
import datetime
import hashlib
import hmac
import json
import urllib
import urllib.parse
import urllib.request
import requests



class HuobiClient(object):

    MARKET_URL = "https://api.huobi.pro"
    TRADE_URL = "https://api.huobi.pro"
    
    def __init__(self, access_key, secret):
        self.access_key = access_key
        self.secret = secret
    
    def http_get_request(self, url, params, add_to_headers=None):
        headers = {
            "Content-type": "application/x-www-form-urlencoded",
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
        }
        if add_to_headers:
            headers.update(add_to_headers)
        postdata = urllib.parse.urlencode(params)
    
        try:
            response = requests.get(url, postdata, headers=headers, timeout=5)
    
            if response.status_code == 200:
                if response.json().get("status", "ok") != "ok":
                    raise ValueError("Error while downloading data for %s where data is %s" % (url, str(response.json())))
                return response.json()
            else:
                return
        except BaseException as e:
            print("httpGet failed, detail is:%s,%s" %(response.text,e))
            return
    
    
    def http_post_request(self, url, params, add_to_headers=None):
        headers = {
            "Accept": "application/json",
            'Content-Type': 'application/json'
        }
        if add_to_headers:
            headers.update(add_to_headers)
        postdata = json.dumps(params)
    
        try:
            response = requests.post(url, postdata, headers=headers, timeout=10)
            if response.status_code == 200:
                if response.json().get("status", "ok") != "ok":
                    raise ValueError("Error while downloading data for %s where data is %s" % (url, str(response.json())))
                return response.json()
            else:
                return
        except BaseException as e:
            print("httpPost failed, detail is:%s,%s" %(response.text,e))
            return
    
    
    def api_key_get(self, params, request_path):
        method = 'GET'
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        params.update({'AccessKeyId': self.access_key,
                       'SignatureMethod': 'HmacSHA256',
                       'SignatureVersion': '2',
                       'Timestamp': timestamp})
    
        host_url = self.TRADE_URL
        host_name = urllib.parse.urlparse(host_url).hostname
        host_name = host_name.lower()
        params['Signature'] = self.createSign(params, method, host_name, request_path, self.secret)
    
        url = host_url + request_path
        return self.http_get_request(url, params)
    
    
    def api_key_post(self, params, request_path):
        method = 'POST'
        timestamp = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
        params_to_sign = {'AccessKeyId': self.access_key,
                          'SignatureMethod': 'HmacSHA256',
                          'SignatureVersion': '2',
                          'Timestamp': timestamp}
    
        host_url = self.TRADE_URL
        host_name = urllib.parse.urlparse(host_url).hostname
        host_name = host_name.lower()
        params_to_sign['Signature'] = self.createSign(params_to_sign, method, host_name, request_path, self.secret)
        url = host_url + request_path + '?' + urllib.parse.urlencode(params_to_sign)
        return self.http_post_request(url, params)
    
    def get(self, path, params):
        return self.api_key_get(params, path)

    def post(self, path, params):
        return self.api_key_post(params, path)
    
    def createSign(self, pParams, method, host_url, request_path, secret_key):
        sorted_params = sorted(pParams.items(), key=lambda d: d[0], reverse=False)
        encode_params = urllib.parse.urlencode(sorted_params)
        payload = [method, host_url, request_path, encode_params]
        payload = '\n'.join(payload)
        payload = payload.encode(encoding='UTF8')
        secret_key = secret_key.encode(encoding='UTF8')
    
        digest = hmac.new(secret_key, payload, digestmod=hashlib.sha256).digest()
        signature = base64.b64encode(digest)
        signature = signature.decode()
        return signature
