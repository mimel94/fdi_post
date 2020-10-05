import logging

import aiohttp
import aiohttp_jinja2
from aiohttp import web
from faker import Faker
from PIL import Image
import cv2
import tensorflow as tf
import json
import numpy as np
from scipy.misc import imread
import base64
from io import BytesIO
from lib.src.align import detect_face  # for MTCNN face detection
log = logging.getLogger(__name__)
from utils import (
    load_model,
    get_face,
    get_faces_live,
    forward_pass,
    save_embedding,
    load_embeddings,
    identify_face,
    allowed_file,
    remove_file_extension,
    save_image
)

# Load FaceNet model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
model_path = 'model/20170512-110547/20170512-110547.pb'
facenet_model = load_model(model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
image_size = 160
image2Train = ''

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# Initiate persistent FaceNet model in memory
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

# Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)



def get_random_name():
    fake = Faker()
    return fake.name()


async def index(request):
    ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    #if not ws_ready.ok:
    #    return aiohttp_jinja2.render_template('index.html', request, {})

    await ws_current.prepare(request)

    name = get_random_name()
    log.info('%s joined.', name)

    await ws_current.send_json({'action': 'connect', 'name': name})

    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'join', 'name': name})
    request.app['websockets'][name] = ws_current

    while True:
        msg = await ws_current.receive()        

        if msg.type == aiohttp.WSMsgType.text:
            for ws in request.app['websockets'].values():
                if ws is not ws_current:
                    frame = msg.data.split(';base64,')
                    image = Image.open(BytesIO(base64.b64decode(frame[1])))
                    image.save('accept.png', 'PNG')   
                    #image = np.array(image) 
                    #image = image.to_ndarray(format="bgr24")   
                    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
                    faces,_ = get_faces_live(
                        img = image,
                        pnet = pnet,
                        rnet = rnet,
                        onet = onet,
                        image_size = image_size
                    )           
                    #print(len(faces))
                    entrar = True
                    if entrar == False:                            
                        if len(faces) == 1 :
                            embedding = forward_pass(
                                img= faces[0],
                                session=facenet_persistent_session,
                                images_placeholder=images_placeholder,
                                embeddings=embeddings,
                                phase_train_placeholder=phase_train_placeholder,
                                image_size=image_size
                            )      
                            filename = 'embedding'
                            save_embedding(
                                embedding=embedding,
                                filename=filename,
                                embeddings_path=""
                            )    

                    if entrar == True:
                        embedding_dict = load_embeddings()                 
                        if len(faces)==1:                                                                
                            face_embedding = forward_pass(
                                img = faces[0],
                                session= facenet_persistent_session,
                                images_placeholder=images_placeholder,
                                embeddings=embeddings,
                                phase_train_placeholder=phase_train_placeholder,
                                image_size=image_size
                            )                            
                            _, status = identify_face(
                                embedding=face_embedding,
                                embedding_dict=embedding_dict
                            )                                            
                            if status == True:                                                   
                                print('identificado')
                            else:
                                print('No identificado') 

                    await ws.send_json(
                        {'action': 'sent', 'name': name, 'text': msg.data})
        else:
            break

    del request.app['websockets'][name]
    log.info('%s disconnected.', name)
    for ws in request.app['websockets'].values():
        await ws.send_json({'action': 'disconnect', 'name': name})

    return ws_current


async def test(request):
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"test":"OK"}
        ),
    )