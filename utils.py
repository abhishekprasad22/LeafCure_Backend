import cv2
import numpy as np
import io
from PIL import Image
from rembg import remove

# Map of supported colormap names to cv2 constants (Matching your dataset script)
_COLORMAP_MAP = {
    "JET": cv2.COLORMAP_JET,
    # Add others if you trained on them, but JET is default in my script
}

def remove_background_add_white(image_bytes):
    """
    Removes background using AI and replaces it with solid white.
    Returns an OpenCV BGR image ready for further processing.
    """
    # 1. Decode bytes to PIL Image (rembg works best with PIL)
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    
    # 2. Remove background (Result is RGBA with transparent background)
    # The 'alpha_matting' parameter helps with fine edges like leaf serrations
    no_bg_image = remove(input_image, alpha_matting=True)
    
    # 3. Create a solid white background of the same size
    white_bg = Image.new("RGBA", no_bg_image.size, "WHITE")
    
    # 4. Composite the no-bg image ON TOP of the white background
    # usage: source, destination, mask
    white_bg.paste(no_bg_image, (0, 0), no_bg_image)
    
    # 5. Convert to RGB (drop alpha channel)
    final_pil = white_bg.convert("RGB")
    
    # 6. Convert PIL RGB to OpenCV BGR (so your other functions work)
    final_cv2 = cv2.cvtColor(np.array(final_pil), cv2.COLOR_RGB2BGR)
    
    return final_cv2

def read_image_from_bytes(file_bytes):
    """Converts uploaded bytes to an OpenCV image, handling Alpha channels."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError("Could not decode image bytes")

    # If image has alpha channel (4 channels), drop it like in your dataset script
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Ensure it is 3 channel BGR if it came in as grayscale for consistency
    if img.ndim == 2:
         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
         
    return img

def encode_image_to_bytes(img):
    """Converts OpenCV image back to bytes to send via HTTP."""
    _, encoded_img = cv2.imencode('.png', img)
    return io.BytesIO(encoded_img.tobytes())

# --- Transformation Functions (Synced with create_transformed_datasets.py) ---

def to_grayscale(img_bgr):
    """Converts BGR to Grayscale."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def negative_image(img):
    """Inverts the image colors."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return 255 - img

def histogram_equalize_color(img_bgr):
    """
    Equalize luminance channel (Y) in YCrCb to preserve colors.
    Matches logic from create_transformed_datasets.py
    """
    if img_bgr.ndim == 2:
        return cv2.equalizeHist(img_bgr)
    
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

def false_color_map(gray_img, colormap_name="JET"):
    """Applies a colormap. Input should ideally be grayscale."""
    # Ensure input is grayscale before mapping, as per dataset script logic
    if gray_img.ndim == 3:
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_img
        
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    cmap = _COLORMAP_MAP.get(colormap_name.upper(), cv2.COLORMAP_JET)
    colored = cv2.applyColorMap(gray, cmap)
    return colored