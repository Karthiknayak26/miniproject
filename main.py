import cv2
import os

# Frame dimensions
frameWidth = 1000   # Frame Width
frameHeight = 480   # Frame Height

# Load the Haar Cascade for Russian plate number detection
plateCascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml");
minArea = 500

# Initialize the video capture
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
count = 0

# Get the Downloads folder path
downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")

while True:
    success, img = cap.read()

    if not success:
        break

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect number plates
    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "NumberPlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            imgRoi = img[y:y + h, x:x + w]

            enlargedRoi = cv2.resize(imgRoi, (w * 2, h * 2))  # Adjust the scaling factor as needed
            cv2.imshow("Number Plate", enlargedRoi)
        
    cv2.imshow("Result", img)

    # Save the detected number plate image when 's' is pressed
    if cv2.waitKey(1) & 0xFF == ord('s'):
        file_path = os.path.join(downloads_folder, f"NumberPlate_{count}.jpg")
        cv2.imwrite(file_path, enlargedRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
