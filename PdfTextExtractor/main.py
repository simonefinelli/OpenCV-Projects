import cv2
import pytesseract
from pdf2image import convert_from_path
import os

PDF_ROOT_PATH = os.path.join("PDFs")
PDF_PATH = os.path.join(PDF_ROOT_PATH, "file1.pdf")

def pdf_to_images(pdf_path, output_folder="temp_images"):
    os.makedirs(output_folder, exist_ok=True)
    images = convert_from_path(pdf_path)
    image_paths = []
    
    for i, image in enumerate(images):
        image_path = os.path.join(output_folder, f"page_{i+1}.png")
        image.save(image_path, "PNG")
        image_paths.append(image_path)
    
    return image_paths

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return image

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def extract_text_from_pdf(pdf_path):
    images = pdf_to_images(pdf_path)
    extracted_text = ""
    
    for image_path in images:
        preprocessed_image = preprocess_image(image_path)
        text = extract_text_from_image(preprocessed_image)
        extracted_text += text + "\n"
        os.remove(image_path)  # clean up temporary images
    
    return extracted_text

if __name__ == "__main__":
    pdf_file = PDF_PATH
    text_output = extract_text_from_pdf(pdf_file)
    print("Extracted Text:")
    print(text_output)