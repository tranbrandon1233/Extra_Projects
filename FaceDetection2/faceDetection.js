
const script = document.createElement('script');
script.src = 'https://docs.opencv.org/4.5.4/opencv.js';
document.head.appendChild(script);

// Function to detect faces and save the output
function detectFaces(imagePath, filename) {
    // Read the image using OpenCV
    let mat = cv.imread(imagePath);

    // Convert the image to grayscale
    let gray = new cv.Mat();
    cv.cvtColor(mat, gray, cv.COLOR_RGBA2GRAY);

    // Load the face cascade classifier
    let faceCascade = new cv.CascadeClassifier();
    faceCascade.load('face_cascade.xml'); // Make sure to include the cascade classifier file in your project

    // Detect faces
    let faces = new cv.RectVector();
    faceCascade.detectMultiScale(gray, faces);

    // Draw rectangles around the faces
    for (let i = 0; i < faces.size(); i++) {
        let face = faces.get(i);
        let point1 = new cv.Point(face.x, face.y);
        let point2 = new cv.Point(face.x + face.width, face.y + face.height);
        cv.rectangle(mat, point1, point2, [0, 0, 255, 255], 2);
    }

    // Save the output
    let outputFilename = filename.split('.')[0] + '_faces.' + filename.split('.')[1];
    let outputPath = path.join('images', outputFilename);
    cv.imwrite(outputPath, mat);

    // Release memory
    mat.delete();
    gray.delete();
    faceCascade.delete();
    faces.delete();
}


const app = express();

const upload = multer({ dest: 'uploads/' });

app.post('/upload', upload.single('image'), (req, res) => {
    detectFaces(req.file.path, req.file.originalname);
    res.send('Face detection complete!');
});

app.listen(3000, () => {
    console.log('Server started on port 3000');
});