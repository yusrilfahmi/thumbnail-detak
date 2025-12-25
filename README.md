# Smart News Image Resizer 🖼️

A Streamlit web application for processing news images to perfect **1920x1080 (16:9)** resolution with intelligent face detection and smart cropping.

## Features

- 🔗 **Direct Image URL Support** - Paste any image URL to process
- 📸 **Instagram Integration** - Extract and process images from Instagram posts
- 👤 **Smart Face Detection** - Uses OpenCV Haar Cascade to detect faces
- ✂️ **Intelligent Cropping** - Positions detected faces at rule of thirds
- 📝 **Custom Titles** - Set custom filenames for downloads
- 📥 **PNG Download** - Download processed images in high quality

## Tech Stack

- **Framework:** Streamlit
- **Image Processing:** OpenCV (headless), NumPy, Pillow
- **Network:** Requests

## Local Development

### Prerequisites
- Python 3.9+

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/thumbnail-detak.git
cd thumbnail-detak

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Deployment to Streamlit Cloud

### Step 1: Create GitHub Repository

1. Go to [GitHub](https://github.com) and sign in
2. Click **"+"** → **"New repository"**
3. Name it `thumbnail-detak` (or any name you prefer)
4. Set to **Public** (required for free Streamlit Cloud)
5. Click **"Create repository"**

### Step 2: Push Code to GitHub

Open terminal in your project folder and run:

```bash
# Initialize git (if not already)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Smart News Image Resizer"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/thumbnail-detak.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Select your repository: `YOUR_USERNAME/thumbnail-detak`
5. Branch: `main`
6. Main file path: `app.py`
7. Click **"Deploy!"**

### Step 4: Wait for Deployment

- Streamlit will install dependencies from `requirements.txt`
- Once complete, you'll get a public URL like: `https://your-app-name.streamlit.app`

## File Structure

```
thumbnail-detak/
├── app.py              # Main application
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Usage

1. **Direct Image URL:**
   - Paste any image URL
   - Click "Process Image"
   - Download the result

2. **Instagram Post:**
   - Paste Instagram post URL (e.g., `https://www.instagram.com/p/XXXXX/`)
   - Click "Fetch Instagram Images"
   - Select an image from the carousel
   - Download the result

3. **Custom Title:**
   - Enter a title in the "Title" field
   - The download filename will use this title
   - Leave empty for default "crop 1920x1080"

## Notes

- Instagram extraction only works with **public posts**
- For best results, use high-resolution source images
- The largest detected face is used as the main subject

## License

MIT License
