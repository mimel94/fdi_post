import logging
import redis
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
from datetime import datetime, date, time, timedelta

# Load FaceNet model and configure placeholders for forward pass into the FaceNet model to calculate embeddings
model_path = 'model/20170512-110547/20170512-110547.pb'
facenet_model = load_model(model_path)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger("pc")
image_size = 160
image2Train = ''
global lista_asistencia
global total_muestas_asistencia
global seg_tick
global datetime_inicio
global datetime_fin
lista_asistencia = list()

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

# Initiate persistent FaceNet model in memory
facenet_persistent_session = tf.Session(graph=facenet_model, config=config)

# Create Multi-Task Cascading Convolutional (MTCNN) neural networks for Face Detection
pnet, rnet, onet = detect_face.create_mtcnn(sess=facenet_persistent_session, model_path=None)

rutaHarr=''
global client
client = redis.Redis(host='localhost', port= 6379)

class Alumno:
    id = ''
    asistencia = 0

def registrar_alumno(id):
    a = Alumno()
    a.id = id
    if a.id not in lista_asistencia:
        lista_asistencia.append(a)       

async def index(request):   
    return aiohttp_jinja2.render_template('index.html', request,{})

async def estado(request):
    params = await request.post()    
    rutaHaar = RUTA_ABS + '/vision/embeddings/'+ str(params['id_usuario']) 
    if os.path.exists(rutaHaar+'/'+str(params['id_usuario'])+'.npy'):
        #await ws.send_str("{'accion':'entrenado'}")    
        mensaje = {'accion':'iniciando','status':'1'}
    else:
        mensaje = {'accion':'iniciando','status':'0'}

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            mensaje
        ),
    ) 
   
async def entrenar(request):     
    params = await request.post()     
    rutaHaar = 'vision/embeddings/'+ str(params['id_usuario']) 
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
        filename = str(params['id_usuario'])        
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
    embedding_dict = load_embeddings(params['id_usuario'])                                   
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

async def inicio_asistencia(request):    
    params = await request.post()        
    seg_tick = params['seg_tick']    
    datetime_inicio =datetime.strptime(params['datetime_inicio'], "%Y-%m-%d %H:%M:%S")     
    datetime_fin = datetime.strptime(params['datetime_fin'], "%Y-%m-%d %H:%M:%S")     
    diferenciaHora = datetime_fin.hour - datetime_inicio.hour    
    diferenciaMin = datetime_fin.minute - datetime_inicio.minute   
    segundosClase = (diferenciaHora * 60 + diferenciaMin) * 60   
    total_muestas_asistencia =  int(segundosClase/int(seg_tick))
    room = '{"total_muestras":"'+str(total_muestas_asistencia)+'" }'  
    client.hset( str(params['id_room']), "data", room)
    #estudiantes = params['lista_estudiantes']
    estudiantes = ["69","70","71"]
    client.rpush("lista_"+params['id_room'], *estudiantes )
    return(web.Response(
        content_type="application/json",
        text=json.dumps(
            total_muestas_asistencia                      
        ),       
    ))

async def terminar_asistencia(request):    
    params = await request.post()
    client.delete(str(params['id_room']))        
    client.delete("lista_"+params['id_room'])
    client.delete("muestras"+params['id_usuario'])  
    mensaje = "llaves eliminadas con exito"
    return(web.Response(
        content_type="application/json",
        text=json.dumps(
            mensaje            
        ),       
    ))

async def muestra_asistencia(request):    
    params = await request.post()
    frame = params['data'].split(';base64,')    
    image = Image.open(BytesIO(base64.b64decode(frame[1])))
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    faces,_ = get_faces_live(
        img = image,
        pnet=pnet,
        rnet=rnet,
        onet=onet,
        image_size=image_size
    )
    embedding_dict = load_embeddings(params['id_usuario'])
    for face in faces:
        face_embedding = forward_pass(
            img = face,
            session = facenet_persistent_session,
            images_placeholder = images_placeholder,
            phase_train_placeholder=phase_train_placeholder,
            embeddings=embeddings,
            image_size=image_size
        )
        _, status = identify_face(
            embedding=face_embedding,
            embedding_dict=embedding_dict
        )        
        if status == True:                        
            lista_asistencia = client.lrange("lista_"+params['id_room'], 0, -1) 
            print(">lista asistencia",lista_asistencia)
            for estudiante in lista_asistencia:
                if estudiante.decode("utf-8").lower() == params['id_usuario'].lower():
                    try:                        
                        contador_asistencia = client.get("muestras"+params['id_usuario'])
                        contador_asistencia = int(contador_asistencia.decode("utf-8"))
                        print("contador antes del if:",contador_asistencia)
                    except:
                        pass
                    if not contador_asistencia:                                                
                        client.set("muestras"+params['id_usuario'], 1)                                
                    else:                    
                        total_muestras = client.hget(params['id_room'], 'data').decode('utf-8')                        
                        total_muestras = json.loads(total_muestras)
                        if contador_asistencia < int(total_muestras['total_muestras']):                        
                            contador_asistencia+=1
                            client.set("muestras"+params['id_usuario'],contador_asistencia)
                            print("contador despues del if:",contador_asistencia)                     
                else:
                    print("no entro en la comparacion") 
        else:
            print("no identificado")
        
        return(web.Response(
        content_type="application/json",
        text=json.dumps(
            contador_asistencia             
        ),       
    ))

async def reporte_asistencia(request):
    params = await request.post()
    total_muestras = client.hget(params['id_room'], 'data').decode('utf-8')                        
    total_muestras = json.loads(total_muestras)

    try:
        contador_asistencia = client.get("muestras"+params['id_usuario'])
        contador_asistencia = int(contador_asistencia.decode("utf-8"))
        porcentaje_asistencia = contador_asistencia * 100 / int(total_muestras['total_muestras'])  
    except:
        pass
    if not contador_asistencia:                                                
        mensaje = "El porcentaje de sistencia es: 0%"
    else:         
        mensaje = "El porcentaje de sistencia es: "+str(int(porcentaje_asistencia))+"%"

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