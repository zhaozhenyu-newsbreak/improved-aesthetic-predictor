from sanic import Sanic
from sanic.response import json as json_response
import logging
import traceback

from simple_inference import Predictor

app = Sanic("aesthetic-score")
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
predictor = Predictor()


@app.route('/health', methods=['GET'])
async def health_check(request):
    return json_response({"success": True})


@app.route('/predict', methods=['POST', 'GET'])
async def predict(request):
    result = {'status': 'failed'}
    try:
        if request.method == 'POST':
            doc = request.json
        else:
            doc = request.args
        logging.info("request: {}".format(doc))
        image_url = doc.get('url')
        if not image_url:
            logging.error("image url not provided")
            result['msg'] = "image url not provided"
            return json_response(result)
        normalize = doc.get('normalize', True)
        normalize = str(normalize).lower() == 'true'
        score = await predictor.predict_img_url_async(image_url, normalize=normalize)
        if score is not None:
            result['status'] = 'success'
            result['score'] = score
            return json_response(result)
        else:
            result['msg'] = "image url not accessible"
            return json_response(result)
    except Exception as e:
        msg = str(traceback.format_exc())
        logging.error("error: {}".format(msg))
        result['msg'] = msg
        return json_response(result)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9145)