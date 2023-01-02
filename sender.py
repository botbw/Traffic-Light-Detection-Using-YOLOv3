import urllib.request
import json
import os
import ssl
import numpy as np
import matplotlib.pyplot as plt
import cv2

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

# Request data goes here
# The example below assumes JSON formatting which may be updated
# depending on the format your endpoint expects.
# More information can be found here:
# https://docs.microsoft.com/azure/machine-learning/how-to-deploy-advanced-entry-script
data = {}

# file_img = open('/Users/haoxuanwang/Downloads/remote_test.jpeg', 'rb') 
file_img = open('preview_images/test3.jpeg', 'rb') 

data_str = file_img.read().hex() 

data_arr = np.fromstring(bytes.fromhex(data_str), np.uint8) 
im0 = cv2.imdecode(data_arr, cv2.IMREAD_COLOR)
im0 = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB) 

data = {"image": str(data_str)} 
body = str.encode(json.dumps(data))

url = 'https://navigasion.eastus2.inference.ml.azure.com/score'
# Replace this with the primary/secondary key or AMLToken for the endpoint
api_key = 'kwTPY0LbPgzumjod4tCdy8WfPVFGPf06'
if not api_key:
    raise Exception("A key should be provided to invoke the endpoint")

# The azureml-model-deployment header will force the request to go to a specific deployment.
# Remove this header to have the request observe the endpoint traffic rules
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'traffic-light' }

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()

    result = json.loads(result)
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(error.read().decode("utf8", 'ignore'))