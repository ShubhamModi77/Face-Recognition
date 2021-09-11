import cv2
import face_recognition

imgRay = face_recognition.load_image_file('ImagesBasics/Ray.jpg')
imgElon = cv2.cvtColor(imgRay, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('ImagesBasics/Ray Test.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRay)[0]
encodeRay = face_recognition.face_encodings(imgRay)[0]
cv2.rectangle(imgRay, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
results = face_recognition.compare_faces([encodeRay],encodeTest)
faceDis = face_recognition.face_distance([encodeRay],encodeTest)
print(results,faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0],2)}', (50 ,50), cv2.FONT_HERSHEY_PLAIN, 1, (2, 0, 255) ,2)

cv2.imshow('Ray', imgRay)
cv2.imshow('Ray Test', imgTest)
cv2.waitKey(0)
