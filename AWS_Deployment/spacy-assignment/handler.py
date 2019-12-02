print('container start')
try:
  import unzip_requirements
except ImportError:
  pass
print('unzipped')

import json
import en_core_web_sm
from spacy import displacy
print('imports end')

MODEL = en_core_web_sm.load()
print('model loaded')


def create_ner_spans(text):
    doc = MODEL(text)
    spans = list()
    for token in doc.ents:
        span = {'start': token.start_char,
                'end'  : token.end_char,
                'type' : token.label_
                }
        spans.append(span)
    return spans

def handle_request(event, context):
    request_body = event['body']
    text = json.loads(request_body)['text']
    print(text)
    
    spans = list()

    if text is not None:
        spans = create_ner_spans(text)

    print("spans :", spans)

    body = {
        'spans': spans
    }
    # print(body)
    response = {
        "statusCode": 200,
        "body": json.dumps(body),
        "headers" : {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*'
        }
    }
    return response
