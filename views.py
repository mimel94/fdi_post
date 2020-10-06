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
from settings import *
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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger("pc")
image_size = 160
image2Train = ''

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# Initiate persistent FaceNet model in memory
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

# Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

rutaHarr=''


async def index(request):   
    return aiohttp_jinja2.render_template('index.html', request,{})

def estado(request):
    params = await request.post()    
    rutaHaar = RUTA_ABS + '/vision/embeddings/'+ str(params['usuario']) 
    if os.path.exists(rutaHaar+'/'+str(params['usuario'])+'.npy'):
        #await ws.send_str("{'accion':'entrenado'}")    
        mensaje = {'accion':'iniciando','status':'1'}
    else:
        mensaje = {'accion':'iniciando','status':'0'}"

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            mensaje
        ),
    ) 
   
async def entrenar(request):     
    params = await request.post()     
    rutaHaar = RUTA_ABS + '/vision/embeddings/'+ str(params['usuario']) 
    frame = params['data'].split(';base64,')
    image = Image.open(BytesIO(base64.b64decode(frame[1])))
    #image.save('accept.png', 'PNG')     
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
        filename = str(params['usuario'])        
        save_embedding(
            embedding=embedding,
            filename=filename,
            embeddings_path=rutaHaar
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
    frame = params['data'].split(';base64,')
    image = Image.open(BytesIO(base64.b64decode(frame[1])))  
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    faces,_ = get_faces_live(
        img = image,
        pnet = pnet,
        rnet = rnet,
        onet = onet,
        image_size = image_size
    )     
    embedding_dict = load_embeddings(params['usuario'])                                   
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