import cv2

body_classifier = cv2.CascadeClassifier('C:/Users/Ramon Chettiar/OneDrive/Desktop/Whitehat Jr/PROJECTS/PRO-106/haarcascade_fullbody.xml')

cap = cv2.VideoCapture('C:/Users/Ramon Chettiar/OneDrive/Desktop/Whitehat Jr/PROJECTS/PRO-106/walking.avi')

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
