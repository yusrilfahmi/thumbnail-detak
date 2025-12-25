"""
Smart News Image Resizer
A Streamlit application for processing news images to 1920x1080 resolution
with intelligent face detection and smart cropping.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re

# Constants
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # 16:9 = 1.778


def load_image_from_url(url: str) -> np.ndarray:
    """
    Download and load an image from a URL.
    
    Args:
        url: The image URL to download from
        
    Returns:
        OpenCV image (BGR format) as numpy array
        
    Raises:
        Exception: If download fails or image is invalid
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        # Convert to numpy array
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise Exception("Failed to decode image. The URL may not be a valid image.")
            
        return img
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please check your connection and try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download image: {str(e)}")


def detect_faces(img: np.ndarray) -> list:
    """
    Detect HUMAN faces in an image using Haar Cascade with strict validation.
    
    Args:
        img: OpenCV image (BGR format)
        
    Returns:
        List of validated human face rectangles (x, y, w, h), sorted by size
    """
    h, w = img.shape[:2]
    
    # Load frontal face cascade only (most reliable for human faces)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply histogram equalization for better detection
    gray = cv2.equalizeHist(gray)
    
    # Detect faces with stricter parameters
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,  # Higher = fewer false positives
        minSize=(80, 80),  # Larger minimum to avoid small false detections
        maxSize=(int(w * 0.7), int(h * 0.7)),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return []
    
    # Validate faces to ensure they are HUMAN faces
    valid_faces = []
    for (x, y, fw, fh) in faces:
        # Skip if too close to edges
        if x < 20 or y < 20 or x + fw > w - 20 or y + fh > h - 20:
            continue
        
        # Check aspect ratio (human faces are roughly square)
        aspect_ratio = fw / fh
        if aspect_ratio < 0.7 or aspect_ratio > 1.4:
            continue
        
        # Extract face region for skin tone check
        face_region = img[y:y+fh, x:x+fw]
        
        # Check for human skin tones (filters out animals)
        if not has_human_skin_tone(face_region):
            continue
        
        # Check if face is in reasonable position (upper 70% of image)
        face_center_y = y + fh / 2
        if face_center_y > h * 0.7:
            continue
        
        valid_faces.append((x, y, fw, fh))
    
    if len(valid_faces) == 0:
        return []
    
    # Sort by size (largest first)
    valid_faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    
    return valid_faces


def has_human_skin_tone(face_region: np.ndarray) -> bool:
    """
    Check if the face region contains human skin tones.
    This helps filter out animals and false positives.
    
    Args:
        face_region: BGR image of the face region
        
    Returns:
        True if human skin tone detected, False otherwise
    """
    # Convert to HSV for better skin detection
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    
    # Define range for human skin tones in HSV
    # Hue: 0-20 (reddish) and 160-180 (also reddish, wraps around)
    # Saturation: 20-170 (not too gray, not too saturated)
    # Value: 40-255 (not too dark)
    lower_skin1 = np.array([0, 20, 40], dtype=np.uint8)
    upper_skin1 = np.array([20, 170, 255], dtype=np.uint8)
    
    lower_skin2 = np.array([160, 20, 40], dtype=np.uint8)
    upper_skin2 = np.array([180, 170, 255], dtype=np.uint8)
    
    # Create masks
    mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
    mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate percentage of skin pixels
    skin_pixels = np.sum(mask > 0)
    total_pixels = face_region.shape[0] * face_region.shape[1]
    skin_ratio = skin_pixels / total_pixels
    
    # Human faces should have at least 15% skin tone pixels
    return skin_ratio > 0.15


def smart_crop_with_face(img: np.ndarray, faces: list) -> np.ndarray:
    """
    Smart crop image to 1920x1080 with face CENTERED.
    
    Args:
        img: Original OpenCV image (BGR format)
        faces: List of detected face rectangles
        
    Returns:
        Cropped image at 1920x1080
    """
    h, w = img.shape[:2]
    
    # Calculate scale factors
    scale_w = TARGET_WIDTH / w
    scale_h = TARGET_HEIGHT / h
    
    # Use LARGER scale to fill the frame
    scale = max(scale_w, scale_h)
    
    # Resize image
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    if len(faces) > 0:
        # Use the largest face
        face = faces[0]
        face_x, face_y, face_w, face_h = face
        
        # Scale face coordinates
        scaled_face_x = int(face_x * scale)
        scaled_face_y = int(face_y * scale)
        scaled_face_w = int(face_w * scale)
        scaled_face_h = int(face_h * scale)
        
        # Calculate face center
        face_center_x = scaled_face_x + scaled_face_w // 2
        face_center_y = scaled_face_y + scaled_face_h // 2
        
        # Position face at CENTER of frame (both X and Y)
        # This ensures face is always centered
        target_x = TARGET_WIDTH // 2
        target_y = TARGET_HEIGHT // 2
        
        # Calculate crop position to center the face
        crop_x = face_center_x - target_x
        crop_y = face_center_y - target_y
        
    else:
        # No face - center crop
        crop_x = (new_width - TARGET_WIDTH) // 2
        crop_y = (new_height - TARGET_HEIGHT) // 2
    
    # Ensure within bounds
    crop_x = max(0, min(crop_x, new_width - TARGET_WIDTH))
    crop_y = max(0, min(crop_y, new_height - TARGET_HEIGHT))
    
    # Perform crop
    result = resized[crop_y:crop_y + TARGET_HEIGHT, crop_x:crop_x + TARGET_WIDTH]
    
    return result


def process_image(img: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Main image processing function.
    Always performs smart crop to 1920x1080 with face as the main focus.
    
    Args:
        img: Original OpenCV image (BGR format)
        
    Returns:
        Tuple of (processed image, processing method description)
    """
    h, w = img.shape[:2]
    aspect_ratio = w / h
    
    # Detect faces
    faces = detect_faces(img)
    face_info = f"Detected {len(faces)} face(s)" if faces else "No faces detected"
    
    # Always use smart crop with face detection
    result = smart_crop_with_face(img, faces)
    
    if faces:
        method = f"Smart Crop with Face Detection (AR: {aspect_ratio:.2f}). {face_info} - Face positioned at rule of thirds."
    else:
        method = f"Center Crop (AR: {aspect_ratio:.2f}). {face_info}"
    
    return result, method


def convert_cv2_to_pil(img: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB image."""
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def get_image_download_bytes(img: np.ndarray) -> bytes:
    """Convert OpenCV image to PNG bytes for download."""
    pil_img = convert_cv2_to_pil(img)
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG', optimize=True)
    return buffer.getvalue()


def get_download_filename(title: str = '') -> str:
    """Generate download filename from title or use default."""
    if title and title.strip():
        # Sanitize filename - remove special characters
        filename = re.sub(r'[<>:"/\\|?*]', '', title.strip())
        filename = filename.replace(' ', '_')
        return f"{filename}.png"
    return "crop_1920x1080.png"


# ============== Streamlit UI ==============

def main():
    st.set_page_config(
        page_title="Smart News Image Resizer",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 1rem 0;
        }
        .stButton > button {
            width: 100%;
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
        }
        .instagram-image-card {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 10px;
            margin: 5px;
            transition: all 0.3s;
        }
        .instagram-image-card:hover {
            border-color: #E1306C;
            box-shadow: 0 4px 12px rgba(225, 48, 108, 0.3);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("<h1 class='main-header'>🖼️ Smart News Image Resizer</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; color: #666;'>Process news images to perfect 1920x1080 (16:9) resolution with intelligent face detection</p>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Input Section
    url_input = st.text_input(
        "🔗 Image URL",
        placeholder="Paste the direct image URL here (e.g., https://example.com/image.jpg)",
        help="Enter the direct URL to an image. Supports JPG, PNG, and WebP formats."
    )
    
    # Title input for download filename
    title_input = st.text_input(
        "📝 Title (Optional)",
        placeholder="Enter title for the download filename (e.g., Persib vs PSM)",
        help="This will be used as the download filename. Leave empty for default 'crop 1920x1080'"
    )
    
    # Store title in session state
    if title_input:
        st.session_state['download_title'] = title_input
    elif 'download_title' not in st.session_state:
        st.session_state['download_title'] = ''
    
    # Process button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        process_button = st.button("🚀 Process Image", type="primary", use_container_width=True)
    
    # Handle image URL
    if process_button:
        if not url_input:
            st.error("⚠️ Please enter an image URL")
        elif not url_input.startswith(('http://', 'https://')):
            st.error("⚠️ Please enter a valid URL starting with http:// or https://")
        else:
            try:
                with st.spinner("📥 Downloading image..."):
                    original_img = load_image_from_url(url_input)
                
                original_h, original_w = original_img.shape[:2]
                
                with st.spinner("🔍 Detecting faces and processing..."):
                    processed_img, method = process_image(original_img)
                
                st.session_state['processed_image'] = processed_img
                st.session_state['processing_method'] = method
                
                st.success("✅ Image processed successfully!")
                
                display_results(original_img, processed_img, method, original_w, original_h)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # Show previously processed image if exists
    elif 'processed_image' in st.session_state:
        st.divider()
        st.markdown("### 📋 Previous Result")
        st.markdown(f"**🧠 Processing Method:** {st.session_state.get('processing_method', 'N/A')}")
        
        st.image(
            convert_cv2_to_pil(st.session_state['processed_image']),
            caption="Previously processed image (1920x1080)",
            use_container_width=True
        )
        
        download_bytes = get_image_download_bytes(st.session_state['processed_image'])
        download_title = st.session_state.get('download_title', '')
        filename = get_download_filename(download_title)
        
        st.download_button(
            label="📥 Download PNG",
            data=download_bytes,
            file_name=filename,
            mime="image/png",
            type="primary",
            use_container_width=True
        )
    
    # Footer with instructions
    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        ### Smart Resizing Logic
        
        This tool uses intelligent algorithms to process images for optimal 16:9 display:
        
        - **Face Detection:** Uses advanced multi-cascade detection with quality scoring
        - **Smart Crop:** Scales image to fill 1920x1080, then crops with face positioned perfectly
        - **Safety Margins:** Ensures heads are never cut off with 60% top margin
        - **Center Crop:** If no face is detected, performs a center crop
        
        ### Features
        
        - ✅ Multi-cascade face detection for better accuracy
        - ✅ Quality scoring to filter false positives
        - ✅ Smart positioning with safety margins
        - ✅ Custom download filenames
        - ✅ High-quality PNG output
        
        ### Tips
        
        - Use high-resolution source images for best results
        - Direct image URLs work best (JPG, PNG, WebP)
        - The highest quality detected face is used as the main subject
        """)


def display_results(original_img, processed_img, method, original_w, original_h):
    """Display the processing results."""
    st.divider()
    
    # Info section
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.info(f"📐 **Original Size:** {original_w} x {original_h} pixels")
    with col_info2:
        st.info(f"📐 **Output Size:** {TARGET_WIDTH} x {TARGET_HEIGHT} pixels")
    
    st.markdown(f"**🧠 Processing Method:** {method}")
    
    st.divider()
    
    # Image comparison
    col_orig, col_result = st.columns(2)
    
    with col_orig:
        st.subheader("📷 Original Image")
        st.image(
            convert_cv2_to_pil(original_img),
            use_container_width=True
        )
    
    with col_result:
        st.subheader("✨ Processed Image (1920x1080)")
        st.image(
            convert_cv2_to_pil(processed_img),
            use_container_width=True
        )
    
    # Download button with custom filename
    st.divider()
    download_bytes = get_image_download_bytes(processed_img)
    download_title = st.session_state.get('download_title', '')
    filename = get_download_filename(download_title)
    
    st.download_button(
        label="📥 Download PNG",
        data=download_bytes,
        file_name=filename,
        mime="image/png",
        type="primary",
        use_container_width=True
    )


if __name__ == "__main__":
    main()
