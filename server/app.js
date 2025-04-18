const express = require('express');
const multer = require('multer');
const cors = require('cors');
const { exec } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

const app = express();
const port = 3000;

// Middleware
app.use(cors({ origin: true }));
app.use(express.json());
app.use('/dresses', express.static(path.join(__dirname, '../dresses')));
app.use('/output', express.static(path.join(__dirname, '../output')));

// File upload setup
const uploadDir = path.join(__dirname, '../uploads');
fs.mkdir(uploadDir, { recursive: true })
  .then(() => console.log('Uploads directory ready.'))
  .catch(err => console.error('Error creating uploads directory:', err));

const storage = multer.diskStorage({
  destination: uploadDir,
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  }
});
const upload = multer({ storage });

// POST /upload
app.post('/upload', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    const imagePath = req.file.path;
    const filename = path.basename(imagePath, path.extname(imagePath));
    const outputDir = path.join(__dirname, '../output');
    const modelPath = path.join(__dirname, '../models/cloth_segm.pth');
    const featuresPath = path.join(outputDir, 'features', `${filename}_cloth_features.json`);
    const matchesPath = path.join(outputDir, 'matches', `${filename}_cloth_features_matches.json`);
    const segmentedPath = path.join(outputDir, 'extracted', `${filename}_cloth.png`);

    // Run segment.py
    await new Promise((resolve, reject) => {
      exec(
        `python ../scripts/segment.py --image_path "${imagePath}" --output_dir "${outputDir}" --model_path "${modelPath}"`,
        (err, stdout, stderr) => {
          if (err) return reject(stderr);
          console.log(stdout);
          resolve();
        }
      );
    });

    // Run extract.py
    await new Promise((resolve, reject) => {
      exec(
        `python ../scripts/extract.py --png_path "${segmentedPath}" --output_dir "${outputDir}" --device cpu`,
        (err, stdout, stderr) => {
          if (err) return reject(stderr);
          console.log(stdout);
          resolve();
        }
      );
    });

    // Run match.py
    await new Promise((resolve, reject) => {
      exec(
        `python ../scripts/match.py --json_path "${featuresPath}" --uri "mongodb+srv://relove_user:myrelovepass007@relovecluster.3efxwzq.mongodb.net/?retryWrites=true&w=majority&appName=ReloveCluster" --output_dir "${outputDir}"`,
        (err, stdout, stderr) => {
          if (err) return reject(stderr);
          console.log(stdout);
          resolve();
        }
      );
    });

    // Read matches
    const matches = JSON.parse(await fs.readFile(matchesPath));

    // Respond
    res.json({
      segmented: `/output/extracted/${filename}_cloth.png`,
      matches: matches.map(m => ({
        ...m,
        image: `/${m.image}` // Ensure accessible path
      }))
    });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: 'Processing failed: ' + error });
  }
});

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`);
});