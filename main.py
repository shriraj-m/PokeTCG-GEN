import cv2
import pytesseract
import pandas as pd
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load Pokémon data
df = pd.read_csv("pokemon_data.csv")

# Set up camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect edges and find the largest contour (likely the card)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Ignore small objects
            if area > max_area:
                max_area = area
                largest_contour = contour

    detected_name = None
    detected_generation = None

    if largest_contour is not None:
        x, y, w, h = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw card rectangle

        # Crop the name area dynamically (Top 5%-15% of the card)
        name_top = y + int(h * 0.05)
        name_bottom = y + int(h * 0.15)  # Adjusted to use a more dynamic approach
        name_area = frame[name_top:name_bottom, x:x + w]

        # Convert to grayscale
        name_gray = cv2.cvtColor(name_area, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's Thresholding
        ret, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

        # Define a rectangular kernel
        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

        # Apply dilation to expand text blocks
        dilated = cv2.dilate(name_thresh, rect_kernel, iterations=1)

        # Find contours (text blocks)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            # Crop the detected text region
            cropped_name = name_thresh[y:y+h, x:x+w]

            # Apply OCR with improved configuration
            text = pytesseract.image_to_string(cropped_name, config="--oem 3 --psm 7")  # Changed psm for better accuracy

            # Match text with Pokémon names
            for name in df["Name"]:
                if name.lower() in text.lower():
                    detected_name = name
                    detected_generation = df[df["Name"] == name]["Generation"].values[0]
                    print(f"Detected: {detected_name} (Gen {detected_generation})")
                    break
 
        # Display detected Pokémon name
        if detected_name:
            text_x = x
            text_y = max(y - 10, 30)
            cv2.putText(frame, f"{detected_name} (Gen {detected_generation})", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the video feed
    cv2.imshow("Pokemon Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()