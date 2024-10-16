# 540asg
# OCR Image Processing with Tesseract and OpenCV

This project shows how to use Tesseract and OpenCV to perform Optical Character Recognition (OCR) processing on images, including image preprocessing, text extraction, border drawing, and more.

## content
- [install](#install)
- [Instructions](#Instructions)
- [function](#function)
- [example](#example)

## install

First, you need to install Tesseract OCR and the associated Python library.

### install Tesseract OCR and development libraries

bash

sudo apt install tesseract-ocr libtesseract-dev


### install Python 

bash

pip install pytesseract opencv-python


### Download sample images

bash

wget https://nanonets.com/blog/content/images/2019/12/invoice-sample.jpg -O image.jpg

wget https://nanonets.com/blog/content/images/2019/12/greek-thai.png


## Instructions

### Import the necessary libraries and set the image path

python

import cv2

import pytesseract

import os


image_path = ________


### Checks if a file exists and reads the image

python

if not os.path.isfile(image_path):

print(f"Error: The file at {image_path} does not exist.")

else:

try:

img = cv2.imread(image_path)

if img is not None:

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

custom_config = r'--oem 3 --psm 6'

text = pytesseract.image_to_string(img_rgb, config=custom_config)

print(text)

else:

print("Error: Unable to read the image file.")

except Exception as e:

print(f"An error occurred: {e}")


### Image pre-processing

python

def get_grayscale(image):

return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def remove_noise(image):

return cv2.medianBlur(image, 5)


def thresholding(image):

return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def dilate(image):

kernel = np.ones((5, 5), np.uint8)

return cv2.dilate(image, kernel, iterations=1)


def erode(image):

kernel = np.ones((5, 5), np.uint8)

return cv2.erode(image, kernel, iterations=1)


def opening(image):

kernel = np.ones((5, 5), np.uint8)

return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def canny(image):

return cv2.Canny(image, 100, 200)


def deskew(image):

coords = np.column_stack(np.where(image > 0))

angle = cv2.minAreaRect(coords)[-1]

if angle < -45:

angle = -(90 + angle)

else:

angle = -angle

(h, w) = image.shape[:2]

center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, angle, 1.0)

rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

return rotated


def match_template(image, template):

return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)


### Perform OCR using preprocessed images and display the results

python

img = gray

cv2.imshow(img)

custom_config = r'--oem 3 --psm 6'

pytesseract.image_to_string(img, config=custom_config)


### Extract data from an image and get a text box

python

import cv2

import pytesseract

from pytesseract import Output


img = cv2.imread('image.jpg')

d = pytesseract.image_to_data(img, output_type=Output.DICT)


### Draw a border around the text

python

h, w, c = img.shape

boxes = pytesseract.image_to_boxes(img)

for b in boxes.splitlines():

b = b.split(' ')

img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)


cv2.imshow(img)

cv2.waitKey(0)


### Drawing borders around words

python

n_boxes = len(d['text'])

for i in range(n_boxes):

if int(d['conf'][i]) > 60:

(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow(img)

cv2.waitKey(0)


### Text Template Matching

python

import re


date_pattern = '^(0[1-9]|[12][0-9]|3[01])-/.-/.\d\d$'


n_boxes = len(d['text'])

for i in range(n_boxes):

if int(d['conf'][i]) > 60:

if re.match(date_pattern, d['text'][i]):

(x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])

img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow(img)

cv2.waitKey(0)


### Detect digital only

python

custom_config = r'--oem 3 --psm 6 outputbase digits'

print(pytesseract.image_to_string(img, config=custom_config))


### Blacklisted characters

python

custom_config = r'-c tessedit_char_blacklist=0123456789 --psm 6'

pytesseract.image_to_string(img, config=custom_config)


### Select Language

python

custom_config = r'-l eng --psm 6'

pytesseract.image_to_string(img, config=custom_config)


### multilingual detection

python

custom_config = r'-l grc+tha+eng --psm 6'

pytesseract.image_to_string(img, config=custom_config)


### language testing

python

from langdetect import detect_langs


custom_config = r'-l grc+tha+eng --psm 6'

txt = pytesseract.image_to_string(img, config=custom_config)

detect_langs(txt)


### Reading text from an image and looping through pages

python

def read_text_from_image(image):

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

dilation = cv2.dilate(thresh, rect_kernel, iterations=1)

contours, hierachy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

image_copy = image.copy()

for contour in contours:

x, y, w, h = cv2.boundingRect(contour)

cropped = image_copy[y: y + h, x: x + w]

file = open("results.txt", "a")

text = pytesseract.image_to_string(cropped)

file.write(text)

file.write("\n")

file.close()


image = cv2.imread("dateimg.jpg")

read_text_from_image(image)


cv2.imshow(image)

f = open("results.txt", "r")

lines = f.readlines()

lines.reverse()

for line in lines:

print(line)

f.close()


## example

The following are sample images and the corresponding OCR processing results.

### Sample images
![image](https://github.com/user-attachments/assets/69b69e7c-dae4-4d96-962b-0b3f1223ccd3)
![image](https://github.com/user-attachments/assets/36c087ec-2d9a-41a9-bc86-9a33b70cd6ef)

### OCR outcome

![image](https://github.com/user-attachments/assets/335345cb-83ec-43f0-ac78-85776bb621f6)

Sample OCR Result Text

ม ส ส โณ ๕ ร ก BE
1 บะ // ห “AHA S.tw Mahakaruna Dharani FRA EARR
TULL TLERAT HAT CLA TY TARGATA
YHRATAHRA EMRKAFH ETRAL I TLS
THATOHLA GTI LAT ALTA ITH T LAG
AU AIC HHH E HY ESAHTAK EN TAH
TATTHITH LES TEL FHT CALA TAH
ธร ชง อม «๕ ศศ ส ส ุ ย อ สุข ส ส มุ ๕ ม ุ ๕ ม ุ 24%
ἐιξεξεδῖθες9 ἐξ ξ ει Ὑσίξ 7641
4140 ๓ จ ธ ข ๆ 4 ๕ ส 4 จ ส ม ุ ม ศุ ม ร 4 ส 797
ฆ ๆ ฤ ณ (อ ย จ @ ฒ ส ุ ภ ศ ุ ภู ส ชู ก ภ ข ธร ม ร ู กิ
ERE RAT ELC TA EA ETOHATAAT
«ἀξ αςκηπεχτας σὰ 1 σι ἀπ Κις
ม ภ ๆ ซ สม %@ ๕ สุ ข ซ %@ ๕ ส ร 1 ส ει ἐξέ ατας
ERAT M LERCH THAME RA FLRAA
TH ERT FHA TA ERY FARIA EAT THE
TAT THERA T THRE TERT HTL GAA T
HAE THE 4 THERE GH FI UG ACAER
% Bib i$ A A it BS Ὁ 6 ἃ dro AS Pe BE AE δὰ
Kes & BR

