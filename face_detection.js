const script = document.createElement('script');
script.src = 'https://docs.opencv.org/4.5.4/opencv.js';
script.onload = onOpenCvReady;
document.head.appendChild(script);

function onOpenCvReady() {
   const input = document.createElement('input');
   input.type = 'file';
   input.accept = 'image/*';
   input.addEventListener('change', onFileSelected);
   document.body.appendChild(input);

   async function onFileSelected(event) {
       const file = event.target.files[0];
       const reader = new FileReader();

       reader.onload = async function(event) {
           const imgData = cv.imread(new Uint8Array(event.target.result));
           const faces = await detectFaces(imgData);
           displayImageWithFaces(imgData, faces);
           imgData.delete();
       };

       reader.readAsArrayBuffer(file);
   }

   async function detectFaces(imgData) {
       const gray = new cv.Mat();
       cv.cvtColor(imgData, gray, cv.COLOR_RGBA2GRAY, 0);

       const faceCascade = new cv.CascadeClassifier();
       faceCascade.load('face_cascade.xml');

       const faces = new cv.RectVector();
       const scaleFactor = 1.1;
       const minNeighbors = 3;
       faceCascade.detectMultiScale(gray, faces, scaleFactor, minNeighbors, 0, 0);

       gray.delete();
       faceCascade.delete();

       return faces;
   }

   function displayImageWithFaces(imgData, faces) {
       const canvas = document.createElement('canvas');
       const ctx = canvas.getContext('2d');
       canvas.width = imgData.cols;
       canvas.height = imgData.rows;
       cv.imshow(canvas, imgData);

       for (let i = 0; i < faces.size(); ++i) {
           const rect = faces.get(i);
           ctx.strokeStyle = 'red';
           ctx.lineWidth = 2;
           ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
       }

       document.body.appendChild(canvas);
   }
}