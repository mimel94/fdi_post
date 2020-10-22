function main(){  
  let CVD = null; // return of Canvas2DDisplay  

  JEEFACEFILTERAPI.init({
    canvasId: 'jeeFaceFilterCanvas',
    NNCPath: 'neuralNets/NN_4EXPR_0.json', // root of NN_DEFAULT.json file    
    //maxFacesDetected:SETTINGS.maxFaces,
    callbackReady: function(errCode, spec){
      if (errCode){
        console.log('Ocurrio un error, lo chento :( . ERROR =', errCode);
        return;
      }

      console.log('INFO: JEEFACEFILTERAPI ESTA LISTO');
      CVD = JeelizCanvas2DHelper(spec);
      CVD.ctx.strokeStyle = 'yellow';
    },

    // called at each render iteration (drawing loop):
    callbackTrack: function(detectState){      
      
      if (detectState.detected>0.95){
        // draw a border around the face:
        //console.log(detectState.)
        const faceCoo = CVD.getCoordinates(detectState);
        CVD.ctx.clearRect(0,0,CVD.canvas.width, CVD.canvas.height);
        CVD.ctx.strokeRect(faceCoo.x, faceCoo.y, faceCoo.w, faceCoo.h);
        CVD.update_canvasTexture();

        if(detectState.expressions){
          
          CVD.ctx.font = "bold 22px sans-serif";
          const expr = detectState.expressions;
          const mouthOpen = expr[0];
          const mouthSmile = expr[1];
          const eyebrowFrown = expr[2];
          const eyebrowRaised = expr[3];
          if (mouthSmile > 0.5){
            //console.log("feliz");         
            
            //CVD.ctx.fillText("Feliz",faceCoo.x,faceCoo.y);
          }
          else if(mouthOpen > 0.27 || eyebrowRaised >0.2){
            //console.log("sorprendido");      
                                               
            //CVD.ctx.fillText("Sorprendido",faceCoo.x,faceCoo.y);
            
          }
          else if (eyebrowFrown>0.2){
            //console.log("Triste")
            //CVD.ctx.fillText("",0,0);
            //CVD.ctx.fillText("Triste",faceCoo.x,faceCoo.y);
          }        
          }else{
            CVD.ctx.fillText("",0,0);
          }
        
      }else{
        CVD.ctx.clearRect(0,0,CVD.canvas.width, CVD.canvas.height);
      }
      
      
      
      
      
      CVD.draw();
    }
  }); //end JEEFACEFILTERAPI.init call
   /* gazefilter.tracker.init({ // 1
      nnc: 'neuralNets/NNCveryLight.json',
      //nnc: 'neuralNets/NN_4EXPR_0.json',
      puploc: 'model/puploc.bin',
    });
    const canvas = document.getElementById('jeeFaceFilterCanvas');

    gazefilter.visualizer.setCanvas(canvas); // 2  


    gazefilter.tracker.connect();  // 3

    console.log(gazefilter.tracker.detected);*/

    
    

    
    
    

} //end main()