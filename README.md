# PNG to Photoshop Layers Decomposer (Production System)

An AI-powered system that transforms flat advertising creatives (PNG/JPEG) into fully layered, editable Photoshop documents (`.psd`) with native editable Type layers, mathematical vector CTA buttons, neural inpainting, bilingual Hindi + English OCR, exact typography matching, and an Adobe UXP Photoshop panel plugin.

---

## 🌟 Key Features & Neural Pipeline

1. **Bilingual Hindi & English Font Extraction**:
   - Powered by EasyOCR with simultaneous Latin & Devanagari models (`['hi', 'en']`).
   - Extracts exact text without character corruption or missing glyph boxes.
   - Accurate typographic classification:
     - **Poppins** (`Poppins-Black`, `Poppins-Bold`, `Poppins-SemiBold`, `Poppins-Medium`, `Poppins-Regular`) — the industry standard for modern Indian & international ad creatives.
     - **Playfair Display** (`PlayfairDisplay-Bold`, `PlayfairDisplay-Regular`, `PlayfairDisplay-Italic`) — high-contrast editorial serifs.
     - **Montserrat** (`Montserrat-Black`, `Montserrat-Bold`) — bold geometric uppercase titles.
     - **Mukta** & **Khand** — specialized Devanagari headline typefaces.

2. **Zero-Ghosting Dynamic Masking & Neural Inpainting**:
   - Luminance variance scanner (`|L - μ_bg| > 1.8 * σ_bg`) combined with adaptive contour dilation.
   - Captures core letter ink, drop shadows, and outer neon glows so inpainting leaves zero halo or ghost bleeding.
   - **LaMa (Large Mask Inpainting)** ONNX Runtime engine backfills backgrounds seamlessly using Fast Fourier Convolutions (FFC).

3. **Native PSD Vector Shapes for CTAs & Badges**:
   - Detects pill buttons ($R = \text{height} / 2$), rounded rectangles, and stroked containers.
   - Extracts 2-stop linear gradients and stroke borders.
   - Compiles native vector shape paths via `ag-psd` that scale infinitely without pixelation.

4. **Web App with Live Typography Inspector**:
   - Standalone web converter accessible at `/dashboard/upload.html`.
   - Displays real-time breakdown of extracted layers with live Google Fonts rendering, color chips, and font sizes.
   - 1-click **Download PSD** button for instant use in Photoshop.

5. **Strict 4-Tier Photoshop Layer Hierarchy**:
   Every document is organized into 4 standardized Photoshop group folders:
   - **`Text Group`** (Top): Native editable Photoshop Type layers (`T` icon) with calibrated fonts, colors, and line leading.
   - **`CTAs & Badges Group`**: Crisp native vector shapes with gradient fills, strokes, and button sub-text placed above the shape.
   - **`Foreground Elements`**: Isolated people, laptops, and props with anti-aliased soft alpha mattes.
   - **`Background Group`** (Bottom): Clean neural inpainted background plate + hidden original reference layer.

---

## 🐳 Deploying on Coolify with Docker

This project is fully containerized and production-ready for deployment on **[Coolify](https://coolify.io)**.

### Step 1: Push Repository to GitHub / GitLab
Ensure your code (including `backend/app/models/` weights and `backend/fonts/` TTF files) is committed and pushed to your Git repository.

### Step 2: Create a New Resource in Coolify
1. In your Coolify dashboard, select your **Project** and **Environment**.
2. Click **+ New Resource** &rarr; select **Public Repository** or **Private Repository (GitHub App)**.
3. Paste your repository URL and select the `main` (or default) branch.

### Step 3: Configure Build Settings
Coolify gives you two deployment options:

#### Option A: Dockerfile (Default Coolify Settings — Easiest)
- **Build Pack**: `Dockerfile`
- **Base Directory**: `/`
- **Dockerfile Location**: `/Dockerfile` (or `/backend/Dockerfile` — both build from repository root `/`)
- **Ports Exposes**: `8000`

#### Option B: Docker Compose
- **Build Pack**: `Docker Compose`
- **Docker Compose Location**: `/docker-compose.yml`
- Coolify will automatically read port `8000` and mount the persistent volume for outputs.

### Step 4: Environment Variables in Coolify
Under the **Environment Variables** tab in Coolify, configure the following:

| Variable | Recommended Value | Description |
| :--- | :--- | :--- |
| `DECOMPOSER_OCR_ENGINE` | `easyocr` | Primary OCR engine (`easyocr`, `tesseract`) |
| `DECOMPOSER_OCR_LANG` | `hi,en` | Languages supported (Hindi + English) |
| `DECOMPOSER_INPAINT_ENGINE` | `lama` | Neural inpainting engine (`lama`, `opencv`) |
| `DECOMPOSER_SEGMENTATION_ENGINE`| `yunet` | Subject segmentation (`yunet`, `classical`) |
| `DECOMPOSER_PSD_ENGINE` | `ag-psd` | Native Photoshop Type layer compiler |
| `DECOMPOSER_MAX_IMAGE_SIDE` | `4096` | Max image resolution allowed |
| `DECOMPOSER_MAX_UPLOAD_MB` | `64` | Max payload size in megabytes |
| `DECOMPOSER_PUBLIC_BASE_URL` | `https://psd.yourdomain.com` | Your public domain (for asset URLs) |

### Step 5: Persistent Storage (Volumes)
Under **Storages / Persistent Storage** in Coolify, add:
- **Volume / Host Path**: `decomposer-outputs`
- **Mount Path**: `/app/outputs`

*This guarantees that generated `.psd` files and layer assets persist across server restarts and container upgrades.*

### Step 6: Domain & SSL
1. In the **Domains** field, enter your public domain (e.g. `psd.yourdomain.com` or `ai-decomposer.yourcompany.com`).
2. Coolify will automatically provision a free Let's Encrypt SSL certificate and set up reverse proxy routing.
3. Click **Deploy**. Once built, the web application and REST API are live!

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.11
- Node.js 18+ or 20 LTS
- Git

### 1. Clone & Set Up Python Environment
```bash
# Clone the repository
git clone <repo-url>
cd "photoshop plugin/backend"

# Create and activate virtual environment
python -m venv .venv

# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies (CPU PyTorch + AI libraries)
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Set Up Node.js Environment (PSD Compiler)
```bash
cd backend
npm install
```

### 3. Launch Local Development Server
```bash
# From backend directory with venv activated:
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser to access the Web UI.

---

## 🎨 Using the Web Dashboard (No Photoshop Required)

1. Navigate to: `http://localhost:8000` (or your live Coolify domain).
2. Drag and drop any flat creative (`.png` or `.jpg`).
3. Click **Convert Image**.
4. The web dashboard will decompose the image and show:
   - **Extracted Layers & Matched Typography**: Every text layer rendered in its exact Google Font with font tag, size, and color.
   - **Isolated Objects**: Transparent PNG cutouts of subjects.
   - **Inpainted Background**: Clean background plate.
5. Click **Download PSD** to download the Photoshop document.

---

## 🔌 Installing into Adobe Photoshop

### Method A: 1-Click Installer (`.ccx`)
1. Ensure Adobe Photoshop (version 23.3+) and Creative Cloud Desktop are installed.
2. Double-click `dist/PNG_Layer_Decomposer.ccx`.
3. Creative Cloud Desktop will ask: **"Install 'PNG Layer Decomposer'?"** &rarr; Click **Install**.
4. In Photoshop, open **Plugins &rarr; PNG Layer Decomposer &rarr; PNG Decomposer**.

### Method B: Developer Mode (Adobe UXP Developer Tool)
1. Open the **Adobe UXP Developer Tool (UDT)**.
2. Click **Add Plugin** and select `plugin/manifest.json`.
3. In `plugin/index.js`, update the backend endpoint to your live Coolify URL:
   ```javascript
   const BACKEND_URL = "https://psd.yourdomain.com";
   ```
4. Click the menu dots next to the plugin and select **Load**.

---

## 🛠️ REST API Reference

### 1. Decompose Image
`POST /api/v1/decompose-png`
- **Headers**: `Content-Type: multipart/form-data` or `image/png`
- **Body**: File upload or raw image binary
- **Response**:
  ```json
  {
    "job_id": "b913ab2baf8d40a0",
    "width": 1280,
    "height": 1280,
    "psd_url": "http://localhost:8000/assets/b913ab2baf8d40a0/decomposed.psd",
    "engines": {
      "ocr": "easyocr",
      "segmentation": "yunet",
      "inpainting": "lama",
      "design": "opencv-classical",
      "psd": "ag-psd"
    },
    "layers": [
      {
        "id": "text_1",
        "name": "क्या आप थकान और चिंता से परेशान हैं?",
        "type": "text",
        "text": {
          "content": "क्या आप थकान और चिंता से परेशान हैं?",
          "font_family": "Poppins-Bold",
          "font_size_px": 77.3,
          "color_hex": "#FEFE8F",
          "alignment": "left"
        }
      }
    ]
  }
  ```

### 2. Health Check
`GET /api/v1/health`
- **Response**: `{"status": "ok", "version": "1.0.0", "engines_available": {...}}`

### 3. Swagger Documentation
Visit `http://localhost:8000/docs` for interactive OpenAPI specs.

---

## 🧪 Automated Testing

Run the test suite to verify all layer logic:
```bash
cd backend
python -m pytest tests/test_decompose.py -k "not live"
```
All 22 contract, geometry, IoU, inpainting, and layer hierarchy validation tests will execute.
