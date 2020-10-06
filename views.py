import logging

import aiohttp
import aiohttp_jinja2
import os
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



async def index(request):
    '''ws_current = web.WebSocketResponse()
    ws_ready = ws_current.can_prepare(request)
    #if not ws_ready.ok:
    #    return aiohttp_jinja2.render_template('index.html', request, {})

    await ws_current.prepare(request)

    if os.path.exists("embedding.npy"):
        await ws_current.send_str("{'accion':'entrenado'}")
    #await ws_current.send_json({'action': 'connect', 'name': name})
    #data = request.query()
    name = request.query['room']
    print(name)
    request.app['websockets'][name] = ws_current '''     

    return aiohttp_jinja2.render_template('index.html', request,{})

def estado(request):
    if os.path.exists("embedding.npy"):
        mensaje = "{'estado':'1'}" 
    else:
        mensaje = "{'estado':'1'}" 
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            mensaje
        ),
    ) 
   
async def entrenar(request):     
    params = await request.post()  
    print('estoy entrenando...')
    frame = params['data'].split(';base64,')
    image = Image.open(BytesIO(base64.b64decode(frame[1])))
    image.save('accept.png', 'PNG')     
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    faces,_ = get_faces_live(
        img = image,
        pnet = pnet,
        rnet = rnet,
        onet = onet,
        image_size = image_size
    )     
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
        mensaje = "{'accion':'entrenado'}"
    elif len(faces) > 1:
        mensaje = "{'accion':'no_entrenado','razon':'multiples rostros detectados'}"
        print('multiples rostros detectados')
    else:
        mensaje = "{'accion':'no_entrenado','razon':'No se detecto rostro'}"
        print('ningun rostro detectado') 
    

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            mensaje
        ),
    )      

async def validar(request):
    params = await request.post() 
    for key,value in request.query.items():
        print("{}: {}".format(key, value))
    print('estoy validando...')
    print(params['accion'])
    frame = params['data'].split(';base64,')
    image = Image.open(BytesIO(base64.b64decode(frame[1])))
    image.save('accept.png', 'PNG')  
 
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    faces,_ = get_faces_live(
        img = image,
        pnet = pnet,
        rnet = rnet,
        onet = onet,
        image_size = image_size
    )     
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
            mensaje = "{'accion':'comparado','resultado':'1'}"                                             
            print('identificado')
        else:
            mensaje = "{'accion':'comparado','resultado':'0'}"                                             
            print('No identificado')  
         
    elif len(faces) > 1:
        mensaje = "{'accion':'comparado','resultado':'0','error':'multiples rostros detectados'}"                                             
        print('multiples rostros detectados')
    else:
        mensaje = "{'accion':'comparado','resultado':'0','error':'ningun rostro detectado'}"                                             
        print('ningun rostro detectado')
    
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            mensaje
        ),
    )


async def test(request):
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"test":"OK"}
        ),
    )