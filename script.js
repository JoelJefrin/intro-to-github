let uploadedImage = null;

const imageInput = document.getElementById('imageInput');
const processBtn = document.getElementById('processBtn');
const originalCanvas = document.getElementById('originalCanvas');
const hogCanvas = document.getElementById('hogCanvas');
const cellSizeSlider = document.getElementById('cellSize');
const binsSlider = document.getElementById('bins');
const cellSizeValue = document.getElementById('cellSizeValue');
const binsValue = document.getElementById('binsValue');
const featureInfo = document.getElementById('featureInfo');

// Update slider values
cellSizeSlider.addEventListener('input', (e) => {
    cellSizeValue.textContent = e.target.value;
});

binsSlider.addEventListener('input', (e) => {
    binsValue.textContent = e.target.value;
});

// Handle image upload
imageInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                uploadedImage = img;
                displayOriginalImage(img);
                processBtn.disabled = false;
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }
});

// Process button click
processBtn.addEventListener('click', () => {
    if (uploadedImage) {
        const cellSize = parseInt(cellSizeSlider.value);
        const bins = parseInt(binsSlider.value);
        extractHOGFeatures(uploadedImage, cellSize, bins);
    }
});

function displayOriginalImage(img) {
    const ctx = originalCanvas.getContext('2d');
    const maxWidth = 500;
    const scale = Math.min(1, maxWidth / img.width);
    
    originalCanvas.width = img.width * scale;
    originalCanvas.height = img.height * scale;
    
    ctx.drawImage(img, 0, 0, originalCanvas.width, originalCanvas.height);
}

function extractHOGFeatures(img, cellSize, bins) {
    const ctx = originalCanvas.getContext('2d');
    const width = originalCanvas.width;
    const height = originalCanvas.height;
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    
    // Convert to grayscale
    const gray = new Float32Array(width * height);
    for (let i = 0; i < data.length; i += 4) {
        const idx = i / 4;
        gray[idx] = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    }
    
    // Calculate gradients
    const { gradientMag, gradientDir } = calculateGradients(gray, width, height);
    
    // Calculate HOG features
    const hogFeatures = calculateHOG(gradientMag, gradientDir, width, height, cellSize, bins);
    
    // Visualize HOG
    visualizeHOG(hogFeatures, width, height, cellSize, bins);
    
    // Display feature information
    displayFeatureInfo(hogFeatures, cellSize, bins);
}

function calculateGradients(gray, width, height) {
    const gradientMag = new Float32Array(width * height);
    const gradientDir = new Float32Array(width * height);
    
    for (let y = 1; y < height - 1; y++) {
        for (let x = 1; x < width - 1; x++) {
            const idx = y * width + x;
            
            // Sobel operators
            const gx = gray[idx + 1] - gray[idx - 1];
            const gy = gray[idx + width] - gray[idx - width];
            
            gradientMag[idx] = Math.sqrt(gx * gx + gy * gy);
            gradientDir[idx] = Math.atan2(gy, gx);
        }
    }
    
    return { gradientMag, gradientDir };
}

function calculateHOG(gradientMag, gradientDir, width, height, cellSize, bins) {
    const cellsX = Math.floor(width / cellSize);
    const cellsY = Math.floor(height / cellSize);
    const hogFeatures = [];
    
    for (let cy = 0; cy < cellsY; cy++) {
        for (let cx = 0; cx < cellsX; cx++) {
            const histogram = new Array(bins).fill(0);
            
            // Calculate histogram for this cell
            for (let y = 0; y < cellSize; y++) {
                for (let x = 0; x < cellSize; x++) {
                    const px = cx * cellSize + x;
                    const py = cy * cellSize + y;
                    
                    if (px < width && py < height) {
                        const idx = py * width + px;
                        const mag = gradientMag[idx];
                        let angle = gradientDir[idx];
                        
                        // Convert angle to 0-180 degrees (unsigned gradients)
                        angle = (angle + Math.PI) % Math.PI;
                        const binWidth = Math.PI / bins;
                        const binIdx = Math.floor(angle / binWidth) % bins;
                        
                        histogram[binIdx] += mag;
                    }
                }
            }
            
            hogFeatures.push({
                x: cx,
                y: cy,
                histogram: histogram
            });
        }
    }
    
    return hogFeatures;
}

function visualizeHOG(hogFeatures, width, height, cellSize, bins) {
    hogCanvas.width = width;
    hogCanvas.height = height;
    const ctx = hogCanvas.getContext('2d');
    
    // Draw semi-transparent original image
    ctx.globalAlpha = 0.3;
    ctx.drawImage(originalCanvas, 0, 0);
    ctx.globalAlpha = 1.0;
    
    // Draw HOG vectors
    hogFeatures.forEach(cell => {
        const cx = cell.x * cellSize + cellSize / 2;
        const cy = cell.y * cellSize + cellSize / 2;
        
        // Normalize histogram
        const maxVal = Math.max(...cell.histogram);
        if (maxVal === 0) return;
        
        // Draw orientation vectors
        const angleStep = Math.PI / bins;
        ctx.strokeStyle = '#ff0000';
        ctx.lineWidth = 1.5;
        
        for (let i = 0; i < bins; i++) {
            const magnitude = (cell.histogram[i] / maxVal) * (cellSize / 2);
            if (magnitude > 0.1) {
                const angle = i * angleStep;
                const dx = Math.cos(angle) * magnitude;
                const dy = Math.sin(angle) * magnitude;
                
                ctx.beginPath();
                ctx.moveTo(cx - dx, cy - dy);
                ctx.lineTo(cx + dx, cy + dy);
                ctx.stroke();
            }
        }
    });
}

function displayFeatureInfo(hogFeatures, cellSize, bins) {
    const totalCells = hogFeatures.length;
    const featureVectorLength = totalCells * bins;
    
    // Calculate some statistics
    let totalMagnitude = 0;
    hogFeatures.forEach(cell => {
        cell.histogram.forEach(val => totalMagnitude += val);
    });
    
    featureInfo.innerHTML = `
        <h3>HOG Feature Information</h3>
        <p><strong>Total Cells:</strong> ${totalCells}</p>
        <p><strong>Cell Size:</strong> ${cellSize}x${cellSize} pixels</p>
        <p><strong>Orientation Bins:</strong> ${bins}</p>
        <p><strong>Feature Vector Length:</strong> ${featureVectorLength}</p>
        <p><strong>Total Gradient Magnitude:</strong> ${totalMagnitude.toFixed(2)}</p>
        <p><strong>Average Magnitude per Cell:</strong> ${(totalMagnitude / totalCells).toFixed(2)}</p>
    `;
}
