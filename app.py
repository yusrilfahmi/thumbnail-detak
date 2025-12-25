"""
Smart News Image Resizer
A Streamlit application for processing news images to 1920x1080 resolution
with intelligent face detection and smart cropping.
Supports direct image URLs and Instagram post extraction.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import re
import json

# Constants
TARGET_WIDTH = 1920
TARGET_HEIGHT = 1080
TARGET_ASPECT_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # 16:9 = 1.778


def extract_instagram_images(url: str) -> list[str]:
    """
    Extract image URLs from an Instagram post.
    
    Args:
        url: Instagram post URL
        
    Returns:
        List of image URLs from the post
        
    Raises:
        Exception: If extraction fails
    """
    # Clean up the URL
    url = url.strip()
    
    # Validate Instagram URL
    if not re.match(r'https?://(www\.)?instagram\.com/(p|reel)/[\w-]+/?', url):
        raise Exception("Invalid Instagram URL. Please use format: https://www.instagram.com/p/XXXXX/")
    
    # Extract shortcode from URL
    match = re.search(r'instagram\.com/(p|reel)/([\w-]+)', url)
    if not match:
        raise Exception("Could not extract post ID from Instagram URL")
    
    shortcode = match.group(2)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }
    
    image_urls = []
    
    try:
        # Method 1: Try Instagram's embed endpoint
        embed_url = f"https://www.instagram.com/p/{shortcode}/embed/"
        response = requests.get(embed_url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            html_content = response.text
            
            # Extract image URLs from embed page
            # Look for high-res images in the embed HTML
            img_patterns = [
                r'"display_url"\s*:\s*"([^"]+)"',
                r'src="(https://[^"]*cdninstagram\.com[^"]*\.jpg[^"]*)"',
                r'srcset="([^"]*1080[^"]*)"',
            ]
            
            for pattern in img_patterns:
                matches = re.findall(pattern, html_content)
                for match in matches:
                    # Clean up escaped URLs
                    clean_url = match.replace('\\u0026', '&').replace('\\/', '/')
                    if clean_url.startswith('http') and 'cdninstagram' in clean_url:
                        if clean_url not in image_urls:
                            image_urls.append(clean_url)
        
        # Method 2: Try the public API endpoint
        if not image_urls:
            api_url = f"https://www.instagram.com/p/{shortcode}/?__a=1&__d=dis"
            response = requests.get(api_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Navigate the JSON structure
                    if 'graphql' in data:
                        media = data['graphql']['shortcode_media']
                    elif 'items' in data:
                        media = data['items'][0]
                    else:
                        media = data
                    
                    # Check for carousel (multiple images)
                    if 'edge_sidecar_to_children' in media:
                        edges = media['edge_sidecar_to_children']['edges']
                        for edge in edges:
                            node = edge['node']
                            if 'display_url' in node:
                                image_urls.append(node['display_url'])
                    elif 'display_url' in media:
                        image_urls.append(media['display_url'])
                    elif 'image_versions2' in media:
                        candidates = media['image_versions2']['candidates']
                        if candidates:
                            # Get highest quality
                            best = max(candidates, key=lambda x: x.get('width', 0))
                            image_urls.append(best['url'])
                except (json.JSONDecodeError, KeyError):
                    pass
        
        # Method 3: Scrape the main page as fallback
        if not image_urls:
            main_url = f"https://www.instagram.com/p/{shortcode}/"
            response = requests.get(main_url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                html_content = response.text
                
                # Look for og:image meta tag
                og_match = re.search(r'property="og:image"\s+content="([^"]+)"', html_content)
                if og_match:
                    image_urls.append(og_match.group(1))
                
                # Look for images in script tags
                script_pattern = r'"display_url"\s*:\s*"([^"]+)"'
                matches = re.findall(script_pattern, html_content)
                for match in matches:
                    clean_url = match.replace('\\u0026', '&').replace('\\/', '/')
                    if clean_url not in image_urls:
                        image_urls.append(clean_url)
        
        if not image_urls:
            raise Exception("Could not extract images from Instagram post. The post may be private or Instagram is blocking requests.")
        
        return image_urls
        
    except requests.exceptions.Timeout:
        raise Exception("Request timed out while fetching Instagram post.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to fetch Instagram post: {str(e)}")


def is_instagram_url(url: str) -> bool:
    """Check if the URL is an Instagram post URL."""
    return bool(re.match(r'https?://(www\.)?instagram\.com/(p|reel)/[\w-]+/?', url))


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
    Detect faces in an image using Haar Cascade.
    
    Args:
        img: OpenCV image (BGR format)
        
    Returns:
        List of detected face rectangles (x, y, w, h), sorted by area (largest first)
    """
    # Load Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces with multiple scale factors for better detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    if len(faces) == 0:
        return []
    
    # Sort by area (largest first) - prioritize the main subject
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    
    return faces


def smart_crop_with_face(img: np.ndarray, faces: list) -> np.ndarray:
    """
    Smart crop image to 1920x1080 with face as the main focus.
    
    The image is first scaled to ensure it can fill 1920x1080,
    then cropped around the detected face (or center if no face).
    
    Args:
        img: Original OpenCV image (BGR format)
        faces: List of detected face rectangles
        
    Returns:
        Cropped image at 1920x1080
    """
    h, w = img.shape[:2]
    
    # Calculate scale factors for width and height
    scale_w = TARGET_WIDTH / w
    scale_h = TARGET_HEIGHT / h
    
    # Use the LARGER scale to ensure image fills the target area
    # This prevents any empty space in the output
    scale = max(scale_w, scale_h)
    
    # Resize image
    new_width = int(w * scale)
    new_height = int(h * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    # Calculate crop position based on face detection
    if len(faces) > 0:
        # Use the largest face for positioning
        face = faces[0]
        face_x, face_y, face_w, face_h = face
        
        # Scale face position to resized image
        scaled_face_x = int(face_x * scale)
        scaled_face_y = int(face_y * scale)
        scaled_face_w = int(face_w * scale)
        scaled_face_h = int(face_h * scale)
        
        # Calculate face center
        face_center_x = scaled_face_x + scaled_face_w // 2
        face_center_y = scaled_face_y + scaled_face_h // 2
        
        # Position face at rule of thirds (top 1/3 for Y, center for X)
        # This keeps the head from being cut off
        target_face_x = TARGET_WIDTH // 2
        target_face_y = TARGET_HEIGHT // 3
        
        # Calculate crop start position
        crop_x = face_center_x - target_face_x
        crop_y = face_center_y - target_face_y
        
    else:
        # No face detected - center crop
        crop_x = (new_width - TARGET_WIDTH) // 2
        crop_y = (new_height - TARGET_HEIGHT) // 2
    
    # Ensure we don't crop outside the image bounds
    crop_x = max(0, min(crop_x, new_width - TARGET_WIDTH))
    crop_y = max(0, min(crop_y, new_height - TARGET_HEIGHT))
    
    # Perform the crop
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
        "<p style='text-align: center; color: #666;'>Process news images to perfect 1920x1080 (16:9) resolution • Supports direct URLs & Instagram posts</p>",
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Input Section
    url_input = st.text_input(
        "🔗 Image URL or Instagram Post URL",
        placeholder="Paste image URL or Instagram post URL (e.g., https://www.instagram.com/p/XXXXX/)",
        help="Supports direct image URLs (JPG, PNG, WebP) and Instagram post URLs"
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
    
    # Detect URL type and show appropriate button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if url_input and is_instagram_url(url_input):
            fetch_button = st.button("📸 Fetch Instagram Images", type="primary", use_container_width=True)
        else:
            fetch_button = st.button("🚀 Process Image", type="primary", use_container_width=True)
    
    # Handle Instagram URL
    if fetch_button and url_input and is_instagram_url(url_input):
        try:
            with st.spinner("📥 Fetching images from Instagram..."):
                image_urls = extract_instagram_images(url_input)
            
            st.session_state['instagram_images'] = image_urls
            st.session_state['instagram_source'] = url_input
            st.success(f"✅ Found {len(image_urls)} image(s) from Instagram post!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Display Instagram images for selection
    if 'instagram_images' in st.session_state and st.session_state['instagram_images']:
        st.divider()
        st.subheader("📸 Select an Image to Process")
        
        images = st.session_state['instagram_images']
        
        # Create columns for image grid
        cols_per_row = min(len(images), 4)
        cols = st.columns(cols_per_row)
        
        for idx, img_url in enumerate(images):
            with cols[idx % cols_per_row]:
                try:
                    # Load and display thumbnail
                    img_data = load_image_from_url(img_url)
                    pil_img = convert_cv2_to_pil(img_data)
                    
                    st.image(pil_img, caption=f"Image {idx + 1}", use_container_width=True)
                    
                    if st.button(f"✨ Process Image {idx + 1}", key=f"process_{idx}", use_container_width=True):
                        st.session_state['selected_instagram_image'] = img_url
                        st.session_state['selected_image_data'] = img_data
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"Failed to load image {idx + 1}")
    
    # Process selected Instagram image
    if 'selected_image_data' in st.session_state:
        st.divider()
        original_img = st.session_state['selected_image_data']
        original_h, original_w = original_img.shape[:2]
        
        with st.spinner("🔍 Detecting faces and processing..."):
            processed_img, method = process_image(original_img)
        
        st.session_state['processed_image'] = processed_img
        st.session_state['processing_method'] = method
        
        # Clear selection state
        del st.session_state['selected_image_data']
        if 'selected_instagram_image' in st.session_state:
            del st.session_state['selected_instagram_image']
        
        st.success("✅ Image processed successfully!")
        
        # Display results
        display_results(original_img, processed_img, method, original_w, original_h)
    
    # Handle direct image URL
    elif fetch_button and url_input and not is_instagram_url(url_input):
        if not url_input.startswith(('http://', 'https://')):
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
    elif 'processed_image' in st.session_state and 'instagram_images' not in st.session_state:
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
        ### Supported Input Types
        
        **📷 Direct Image URLs:**
        - Paste any direct link to an image (JPG, PNG, WebP)
        - The image will be processed immediately
        
        **📸 Instagram Posts:**
        - Paste an Instagram post URL (e.g., `https://www.instagram.com/p/XXXXX/`)
        - The app will extract all images from the post
        - Select which image you want to process
        - Works with single images and carousel posts
        
        ### Smart Resizing Logic
        
        - **Face Detection:** Uses OpenCV's Haar Cascade to detect faces
        - **Smart Crop:** Scales image to fill 1920x1080, then crops with face at rule of thirds
        - **Center Crop:** If no face is detected, performs a center crop
        
        ### Tips
        
        - For best results, use high-resolution source images
        - Instagram extraction works with public posts
        - The largest detected face is used as the main subject
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
