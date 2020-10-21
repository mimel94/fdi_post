import logging

import jinja2, json


import aiohttp_jinja2
from aiohttp import web
from views import *


async def init_app():

    app = web.Application()

    app['websockets'] = {}

    app.on_shutdown.append(shutdown)    

    app.router.add_get('/aiortc/test', test)
    app.router.add_get('/', index)
    app.router.add_post('/entrenar',entrenar)
    app.router.add_post('/aiortc/validar',validar)
    app.router.add_post('/aiortc/estado',estado)
    app.router.add_post('/asis/iniciar',inicio_asistencia)
    app.router.add_post('/asis/terminar',terminar_asistencia)
    app.router.add_post('/asis/enviar_muestra',muestra_asistencia)
    app.router.add_post('/asis/reporte',reporte_asistencia)
    app.router.add_get('/main.js',javascript_main)
    app.router.add_get('/pupil.js',pupil_tracker)
    app.router.add_get('/gazefilter.js',gazefilter)
    app.router.add_get('/neuralNets/NNCveryLight.json',NNCveryLight)
    app.router.add_get('/model/puploc.bin', model_puploc)
    app.router.add_get('/dist/jeelizFaceFilter.js',javascript_jeelizFaceFilter)
    app.router.add_get('/helpers/Canvas2DDisplay.js',Canvas2DDisplay) 
    #app.router.add_get('/NNC.json',neuralNetsNNC) 
    app.router.add_get('/neuralNets/NN_DEFAULT.json',neuralNets) 
    app.router.add_get('/neuralNets/NN_4EXPR_0.json',neuralNetsExpression) 
    return app


async def shutdown(app):
    for ws in app['websockets'].values():
        await ws.close()
    app['websockets'].clear()


def main():
    logging.basicConfig(level=logging.DEBUG)

    app = init_app()
    web.run_app(app)


if __name__ == '__main__':
    main()
