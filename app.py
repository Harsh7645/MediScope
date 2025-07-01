# # import os
# # os.environ['LD_LIBRARY_PATH'] = "/usr/lib/x86_64-linux-gnu:/lib/x86_64-linux-gnu:" + os.environ.get('LD_LIBRARY_PATH', '')
# # import cv2


# import streamlit as st
# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image

# st.title("MediVision - Medicine Reader")

# uploaded_file = st.file_uploader("Upload medicine image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image', use_column_width=True)
#     img = np.array(image)
#     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     text = pytesseract.image_to_string(thresh)
#     st.subheader("Extracted Text")
#     st.write(text)
#     st.image(thresh, caption='Processed Image', use_column_width=True, clamp=True)

from ai_validation import validate_medicine
from translation import translate_medicine_text
import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import re

# App configuration
st.set_page_config(page_title="MediScope", page_icon="💊")

st.title("💊 MediScope - Medicine Reader")
st.markdown("Upload an image of a medicine package to extract information")

# Medicine database (basic)
medicine_db = {
    "paracetamol": {
        "name": "Paracetamol",
        "use": "Pain reliever and fever reducer",
        "dosage": "Adults: 500-1000mg every 4-6 hours (max 4000mg/day)",
        "warnings": "Do not exceed recommended dose. Avoid alcohol."
    },
    "dolo": {
        "name": "Dolo-650 (Paracetamol)",
        "use": "Pain and fever relief",
        "dosage": "1 tablet every 4-6 hours as needed",
        "warnings": "Maximum 4 tablets in 24 hours"
    },
    "amoxicillin": {
        "name": "Amoxicillin",
        "use": "Antibiotic for bacterial infections",
        "dosage": "Take with food as prescribed by doctor",
        "warnings": "Complete full course even if feeling better"
    }
}

def preprocess_image(img, method="adaptive"):
    """Apply different preprocessing methods"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    if method == "simple":
        # Simple blur only
        processed = cv2.GaussianBlur(gray, (3, 3), 0)
    elif method == "adaptive":
        # Adaptive thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        processed = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
    elif method == "otsu":
        # Otsu thresholding
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, processed = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        processed = gray
    
    return processed

def extract_text_with_configs(image):
    """Try different OCR configurations"""
    configs = [
        r'--oem 3 --psm 6',  # Uniform block of text
        r'--oem 3 --psm 8',  # Single word
        r'--oem 3 --psm 7',  # Single text line
        r'--oem 3 --psm 13'  # Raw line. Treat image as single text line
    ]
    
    results = []
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config).strip()
            if text:
                results.append(text)
        except:
            continue
    
    return results

def find_medicine_info(text):
    """Search for medicine information in extracted text"""
    text_lower = text.lower()
    found_medicines = []
    
    for key, info in medicine_db.items():
        if key in text_lower:
            found_medicines.append(info)
    
    return found_medicines

# File upload
uploaded_file = st.file_uploader(
    "Choose a medicine image", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear image of medicine packaging or label"
)

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    st.subheader("📸 Uploaded Image")
    st.image(image, caption='Original Image', use_container_width=True)
    
    # Convert to OpenCV format
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Preprocessing options
    st.subheader("🔧 Processing Options")
    col1, col2 = st.columns(2)
    
    with col1:
        preprocess_method = st.selectbox(
            "Choose preprocessing method:",
            ["simple", "adaptive", "otsu", "none"],
            index=1,
            help="Different methods work better for different image types"
        )
    
    with col2:
        show_processed = st.checkbox("Show processed image", value=True)
    
    # Process image
    processed_img = preprocess_image(img, preprocess_method)
    
    if show_processed:
        st.subheader("🖼️ Processed Image")
        st.image(processed_img, caption=f'Processed Image ({preprocess_method})', use_container_width=True, clamp=True)
    
    # Extract text with multiple configurations
    st.subheader("📝 Text Extraction")
    
    with st.spinner("Extracting text..."):
        text_results = extract_text_with_configs(processed_img)
    
    if text_results:
        # Show all OCR results
        for i, text in enumerate(text_results):
            if text.strip():
                st.markdown(f"**Result {i+1}:**")
                st.text_area(f"OCR Output {i+1}", text, height=100, key=f"text_{i}")
        
        # Combine all results for medicine search
        combined_text = " ".join(text_results)
        
        # Search for medicine information
        st.subheader("💊 Medicine Information")
        found_medicines = find_medicine_info(combined_text)
        
        if found_medicines:
            for medicine in found_medicines:
                st.success(f"✅ Found: **{medicine['name']}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Use:** {medicine['use']}")
                    st.write(f"**Dosage:** {medicine['dosage']}")
                with col2:
                    st.warning(f"⚠️ **Warning:** {medicine['warnings']}")
        else:
            st.info("No medicine information found in our database. Please consult a healthcare professional.")
            
        # Text-to-speech option
        if st.button("🔊 Read Aloud (Text-to-Speech)"):
            if combined_text.strip():
                try:
                    from gtts import gTTS
                    import io
                    
                    tts = gTTS(text=combined_text, lang='en')
                    mp3_buffer = io.BytesIO()
                    tts.write_to_fp(mp3_buffer)
                    mp3_buffer.seek(0)
                    
                    st.audio(mp3_buffer, format='audio/mp3')
                except ImportError:
                    st.error("Text-to-speech not available. Install gtts: `pip install gtts`")
                except Exception as e:
                    st.error(f"Text-to-speech failed: {e}")
    else:
        st.warning("No text could be extracted from the image. Try a clearer image or different preprocessing method.")

    # OCR text extraction
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_img, config=custom_config)
    
    # Add multilingual support
    translated_text = translate_medicine_text(text)
    
    # AI validation
    is_valid, med_name, confidence = validate_medicine(translated_text)
    
    # Display results
    st.write(f"**Raw OCR:** {text}")
    st.write(f"**Translated:** {translated_text}")

    if is_valid:
        st.success(f"✅ Validated: {med_name} (Confidence: {confidence:.2f})")
        # Add your medicine DB lookup here
    else:
        st.warning("⚠️ AI couldn't verify this medicine. Double-check the label.")

# Sidebar with tips
st.sidebar.title("💡 Tips for Better Results")
st.sidebar.markdown("""
**For best OCR results:**
- Use good lighting
- Keep the image focused
- Avoid shadows or glare
- Try different preprocessing methods
- Crop to show only the label if possible

**Supported medicines:**
- Paracetamol/Dolo-650
- Amoxicillin
- (More coming soon!)
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using Streamlit")
