import cv2 as cv
import  numpy as np
from tensorflow.keras.models import load_model

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
cap = cv.VideoCapture(0)
cascade = cv.CascadeClassifier('Resources/haarcascade_frontalface_default.xml')
model = load_model("C:/Users/Yoga/Desktop/PY/Models/FaceExpressionIdentificaionModel.h5")
print("Model loaded")
while True:
    bool,frame = cap.read()
    if not bool:
        break
    gray = cv.cvtColor(frame,cv.COLOR_RGB2GRAY)
    faces = cascade.detectMultiScale(gray,minNeighbors=5)
    for x,y,w,h in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
        Roiframe = gray[y:y+h,x:x+w]
        Roiframe = cv.resize(Roiframe,(48,48))
        cropimage = np.expand_dims(np.expand_dims(Roiframe,-1),0)
        Detect = model.predict(cropimage)
        #print(Detect)
        maxInd = int(np.argmax(Detect))

        cv.putText(frame,emotion_dict[maxInd],(x+20,y-60),cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),3)
        cv.imshow("Image",frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break