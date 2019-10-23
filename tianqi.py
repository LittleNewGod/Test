import requests
#response = requests.get('https://www.tianqiapi.com/api?version=v1&cityid=101010100')
data= {'version':'v1','cityid':'101010100'}
response = requests.post('https://www.tianqiapi.com/api',data)
response.encoding="utf-8"
#print(response.text)
print (response.json())
print(response.url)


