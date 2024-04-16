import streamlit as st
from PIL import Image
import pre_processing as process
import pytesseract
import numpy as np

# Title for the app
st.title('Text Extraction from Image')

st.header('Steps on preprocessing')

st.subheader('1. Deskewing')
st.write('Deskewing the image is important for accurate text extraction, especially when dealing with images that may not be perfectly aligned. This preprocessing step helps correct any skew or tilt in the image, ensuring that the text recognition algorithms can work effectively regardless of the orientation of the input image. For optimal efficiency, text recognition systems like Tesseract necessitate the bounding box surrounding the text. Therefore, before doing text recognition, this detector may be used to find the bounding boxes.')
st.write('I was unable to use the deskewing in this project but I would have used one of the following methods\n 1. EAST Model [https://github.com/argman/EAST]  \n 2. Craft Text Detector [https://pypi.org/project/craft-text-detector/]')

st.subheader('2. Grayscale Conversion')
st.write("This function converts a color image to grayscale using the OpenCV library's cv2.cvtColor function with the cv2.COLOR_BGR2GRAY parameter. Grayscale conversion is essential for text extraction and analysis as it simplifies the image to a single channel representing intensity.")

st.subheader('3. Image Resizing')
st.write("The Image Resizing function utilizes cv2.resize to double both the height and width of an image. This step is beneficial for standardizing image dimensions or enhancing details, particularly useful for improving text extraction accuracy.")

st.subheader('4. Thresholding')
st.write("The Thresholding function employs Otsu's method (cv2.threshold) to create a binary image, separating text or objects from the background. Binary images are easier to process and analyze, particularly advantageous for text extraction tasks.")

note = 'Note : Implementing noise removal, dilation, erosion, edge detection, and other preprocessing steps can indeed significantly improve the performance of text extraction algorithms. These techniques help in enhancing image quality, reducing noise, highlighting important features, and improving the accuracy of text recognition systems.'

styled_text = f"""
    <div style="
        border: 2px solid #333333;
        border-radius: 5px;
        padding: 10px;
        background-color: #f0f0f0;
        color: black;
        ">
        <p>{note}</p>
    </div>
    """
st.markdown(styled_text, unsafe_allow_html=True)

st.subheader('4. Text Extraction')
st.write('The various text extraction algorithms available are as follows:')
packages = [
    "Tesseract OCR",
    "Pytesseract",
    "Google Cloud Vision API",
    "AWS Rekognition",
    "Microsoft Azure Computer Vision API",
    "OpenCV",
    "PyOCR",
    "OCRopus"
]

# Display the list using st.write
st.write("List of Text Extraction Packages:")
for package in packages:
    st.write(f"- {package}")

st.write("I have used Pytesseract for text extraction in this model")
st.write("Pytesseract can be fine-tuned by modifying its configuration parameters. You can adjust settings such as OCR engine mode, page segmentation mode, language, whitelist/blacklist characters, and more.")
st.write("For example, you can set the OCR engine mode to OEM 1 for LSTM mode, specify the page segmentation mode using the '-psm' parameter, and define the language using the '-l' parameter. Additionally, you can use the '--oem' parameter to specify the OCR engine mode and '--psm' parameter for page segmentation mode.")
st.write("Example command for fine-tuning Pytesseract:")
st.code("pytesseract.image_to_string(image, config='--oem 1 --psm 6 -l eng')")

st.subheader("Comparison of Text Extraction Packages:")
packages = {
    "Tesseract OCR": "Highly accurate and widely used, especially when fine-tuned with appropriate configurations for specific languages and image types.",
    "Google Cloud Vision API": "Known for its accuracy and reliability, providing advanced OCR capabilities along with other image analysis features.",
    "AWS Rekognition": "A cloud-based service offering accurate OCR capabilities, part of the AWS ecosystem and scalable for large-scale text extraction tasks.",
    "Microsoft Azure Computer Vision API": "Known for its accuracy and comprehensive set of image analysis features, including OCR, and integrates well with other Azure services.",
    "OpenCV": "Primarily focused on computer vision tasks but includes functionalities for text extraction using techniques like contour detection and thresholding.",
    "PyOCR": "Provides access to multiple OCR engines, including Tesseract and OCRopus, offering flexibility in choosing the most suitable engine.",
    "OCRopus": "Known for its accuracy, especially in handling complex text layouts and languages, and offers advanced OCR capabilities.",
    "Pytesseract": "Highly accurate when combined with proper preprocessing techniques and configuration adjustments, particularly useful for Python-based projects.",
}

# Displaying the comparison
for package, description in packages.items():
    st.write(f"- **{package}:** {description}")

# Mentioning the preference for Tesseract OCR
st.write("\n If I have to choose one, I will go with **Tesseract OCR**. The accuracy of text extraction heavily depends on preprocessing techniques, and Tesseract provides excellent results when properly fine-tuned.")

st.header("Sample Text Extractor Model")
st.subheader("Please submit a file to extract text")

# Function to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Check if an image is uploaded
if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)

    
    # Display the OpenCV image
    st.image(image, caption='Uploaded Image', use_column_width=False)
    st.write("Image uploaded successfully!")

    gray = process.get_grayscale(np.array(image))

    st.header('Listing different preprocessing steps before extracting text')

    st.subheader('1. GrayScale Image')
    st.image(gray, caption='Gray Scale Image', use_column_width=False)

    resize_image = process.resize_image(gray)

    st.subheader('2. Resize Image')
    st.image(resize_image, caption='Resize Image', use_column_width=False)

    thresh = process.thresholding(resize_image)

    st.subheader('3. Threshold Image')
    st.image(thresh, caption='Threshold Image', use_column_width=False)

    custom_config = r'--psm 6 --oem 3 -l eng'
    text = pytesseract.image_to_string(thresh, config=custom_config)

    st.subheader('OUTPUT :')
    styled_text = f"""
    <div style="
        border: 2px solid #333333;
        border-radius: 5px;
        padding: 10px;
        background-color: #f0f0f0;
        color: darkred;
        ">
        <p>{text}</p>
    </div>
    """
    st.markdown(styled_text, unsafe_allow_html=True)

    # You can perform further processing on the OpenCV image here
    # For example, you can apply image processing algorithms or perform OCR on the image
else:
    # Show a message if no image is uploaded
    st.write('Upload an image using the "Choose File" button above.')
