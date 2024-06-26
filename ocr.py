#python ocr.py "C:\Users\LENOVO\Desktop\projects\ocr\Screenshot 2023-08-28 012834.png" -l en -g 1
from easyocr import Reader
import argparse
from PIL import Image
import cv2
import numpy as np

# Construct the argument parser and parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument("image", help="path to input image")
parser.add_argument("-l", "--langs", type=str, default="en", help="comma separated list of languages for OCR")
parser.add_argument("-g", "--gpu", type=int, default=1, help="whether or not GPU should be used")

args = vars(parser.parse_args())

# Extract the arguments
langs = args["langs"].split(",")
print("[INFO] Using the following languages: {}".format(langs))

# Load the input image from disk
image = cv2.imread(args["image"])
image_pil = Image.open(args["image"]).convert("RGB")
image_pil = np.array(image_pil)

# OCR the input image using EasyOCR
print("[INFO] Performing OCR on input image...")
reader = Reader(lang_list=langs, gpu=args["gpu"] > 0)
results = reader.readtext(image_pil)

# Process and display results
for (bbox, text, prob) in results:
    print("[INFO] {:.4f}: {}".format(prob, text))
    
    # Unpack the bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    tl = (int(top_left[0]), int(top_left[1]))
    br = (int(bottom_right[0]), int(bottom_right[1]))
    
    # Draw bounding box and text on the image
    cv2.rectangle(image, tl, br, (0, 0, 255), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Display the image with bounding boxes
cv2.imshow("Image", image)
cv2.waitKey(0)
