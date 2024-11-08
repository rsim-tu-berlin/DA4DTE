import requests

def predict_with_server(image_path, question):
    url = "http://195.134.71.116:5000/predict"

    files = {'image': open(image_path, 'rb')}
    data = {'string': question}

    response = requests.post(url, files=files, data=data)

    return response.json()

image_path = 'path to .tif image'
question = 'Are there any vessels in this image?'

result = predict_with_server(image_path, question)
print(result)
