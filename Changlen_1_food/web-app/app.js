// app.js
const express = require('express');
const multer = require('multer');
const fetch = require('node-fetch'); // hoặc axios
const fs = require('fs');
const FormData = require('form-data');
const path = require('path');

const app = express();
const upload = multer({ dest: 'uploads/' });

app.set('view engine', 'ejs');
app.use(express.static('public'));

const PY_API = "http://localhost:5001/predict"; // endpoint Flask

app.get('/', (req, res) => {
  res.render('index');
});

app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    const filePath = req.file.path;
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));

    const response = await fetch(PY_API, { method: 'POST', body: form });
    const data = await response.json();

    // xóa file upload tạm
    fs.unlinkSync(filePath);

    res.render('result', { result: data });
  } catch (err) {
    console.error(err);
    res.status(500).send("Error processing image");
  }
});

app.listen(3000, () => console.log(`Express server listening on httpL//localhost:${3000}`));