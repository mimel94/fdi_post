function main(){
  let CVD = null; // return of Canvas2DDisplay

  JEEFACEFILTERAPI.init({
    canvasId: 'jeeFaceFilterCanvas',
    NNCPath: 'neuralNets/', // root of NN_DEFAULT.json file
    callbackReady: function(errCode, spec){
      if (errCode){
        console.log('AN ERROR HAPPENS. SORRY BRO :( . ERR =', errCode);
        return;
      }

      console.log('INFO: JEEFACEFILTERAPI IS READY');
      CVD = JeelizCanvas2DHelper(spec);
      CVD.ctx.strokeStyle = 'yellow';
    },

    // called at each render iteration (drawing loop):
    callbackTrack: function(detectState){
      if (detectState.detected>0.6){
        // draw a border around the face:
        const faceCoo = CVD.getCoordinates(detectState);
        CVD.ctx.clearRect(0,0,CVD.canvas.width, CVD.canvas.height);
        CVD.ctx.strokeRect(faceCoo.x, faceCoo.y, faceCoo.w, faceCoo.h);
        CVD.update_canvasTexture();
        
      }else{
        CVD.ctx.clearRect(0,0,CVD.canvas.width, CVD.canvas.height);
      }
      CVD.draw();
    }
  }); //end JEEFACEFILTERAPI.init call
} //end main()