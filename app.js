const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const { initializeApp } = require("firebase/app");
const { getFirestore, doc, setDoc, collection, getDocs } = require("firebase/firestore");
const cors = require("cors");

// Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAloXmh8M6XojY28ID8TFsSBsSH3msXN3Y",
  authDomain: "submissionmlgc-razanaditya.firebaseapp.com",
  projectId: "submissionmlgc-razanaditya",
  storageBucket: "submissionmlgc-razanaditya.appspot.com",
  messagingSenderId: "616487699689",
  appId: "1:616487699689:web:38f161b04630f139a51c78",
};

// Initialize Firebase
const firebaseApp = initializeApp(firebaseConfig);
const firestore = getFirestore(firebaseApp);

const app = express();
app.use(cors())

// Middleware to limit the file size to 1MB
const upload = multer({
  limits: {
    fileSize: 1000000, // 1MB
  },
}).single("image");

let model;
async function loadModel() {
  model = await tf.loadGraphModel(
    "https://storage.googleapis.com/model-ml-tensor/model-tensor/model.json"
  );
  console.log("Model loaded successfully");
}
loadModel();

// Prediction endpoint
app.post("/predict", (req, res, next) => {
  upload(req, res, async function (err) {
    if (err instanceof multer.MulterError && err.code === "LIMIT_FILE_SIZE") {
      return res.status(413).json({
        status: "fail",
        message: "Payload content length greater than maximum allowed: 1000000",
      });
    }

    if (!req.file) {
      return res.status(400).json({
        status: "fail",
        message: "No image uploaded",
      });
    }

    try {
      // Load the image into TensorFlow.js
      const imageBuffer = req.file.buffer;
      const tensor = tf.node
        .decodeJpeg(imageBuffer)
        .resizeNearestNeighbor([224, 224])
        .expandDims()
        .toFloat();

      // Make prediction
      const prediction = model.predict(tensor);
      const result = prediction.dataSync()[0] > 0.5 ? "Cancer" : "Non-cancer";
      const suggestion =
        result === "Cancer"
          ? "Segera periksa ke dokter!"
          : "Tidak ditemukan penyakit. Tetap jaga kesehatan!";

      // Create a document ID and prepare data
      const id = Math.random().toString(36).substr(2, 9);
      const createdAt = new Date().toISOString();

      // Save the prediction result to Firestore
      await setDoc(doc(firestore, "predictions", id), {
        id: id,
        result: result,
        suggestion: suggestion,
        createdAt: createdAt,
      });

      // Respond with prediction result
      res.status(201).json({
        status: "success",
        message: "Model is predicted successfully",
        data: {
          id: id,
          result: result,
          suggestion: suggestion,
          createdAt: createdAt,
        },
      });
    } catch (error) {
      console.error("Prediction error:", error);
      res.status(500).json({
        status: "fail",
        message: "Terjadi kesalahan dalam melakukan prediksi",
      });
    }
  });
});

// Prediction history endpoint
app.get("/predict/histories", async (req, res) => {
  try {
    // Retrieve all prediction documents from Firestore
    const querySnapshot = await getDocs(collection(firestore, "predictions"));
    const histories = [];

    querySnapshot.forEach((doc) => {
      const data = doc.data();
      histories.push({
        id: data.id,
        history: {
          result: data.result,
          createdAt: data.createdAt,
          suggestion: data.suggestion,
          id: data.id,
        },
      });
    });

    // Respond with the prediction history
    res.json({
      status: "success",
      data: histories,
    });
  } catch (error) {
    console.error("Error retrieving prediction history:", error);
    res.status(500).json({
      status: "fail",
      message: "Terjadi kesalahan dalam mengambil riwayat prediksi",
    });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
