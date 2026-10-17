import os
import fitz  # PyMuPDF

# कॉन्फ़िगरेशन
pdf_filename = "EcoRisk_NJ_Ethylene_Oxide_Cancer_Research_2026.pdf"
output_folder = "Research Paper Images"

# अगर फोल्डर नहीं है तो उसे बनाएं
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if os.path.exists(pdf_filename):
    doc = fitz.open(pdf_filename)
    
    # हाई रेजोल्यूशन (300 DPI) के लिए जूम फैक्टर 3.0 सेट किया गया है ताकि फॉन्ट बिल्कुल न फटें
    zoom = 3.0 
    mat = fitz.Matrix(zoom, zoom)

    print(f"Converting {len(doc)} pages to high-resolution PNG images...")

    for i, page in enumerate(doc):
        pix = page.get_pixmap(matrix=mat)
        image_path = os.path.join(output_folder, f"{i + 1}.png")
        pix.save(image_path)
        print(f"Saved: {image_path}")

    print("✅ Success! All research paper pages converted in crystal-clear high quality.")
else:
    print(f"❌ Error: Could not find PDF file named '{pdf_filename}' in the current directory.")