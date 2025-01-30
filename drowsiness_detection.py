# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer

# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# lbl = 'Open'

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = [99]
# lpred = [99]

# while True:
#     ret, frame = cap.read()
#     height, width = frame.shape[:2]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    
#     for (x, y, w, h) in faces:
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
    
#         left_eye = leye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
#         right_eye = reye_cascade.detectMultiScale(gray[y:y+h, x:x+w])

#         for (ex, ey, ew, eh) in right_eye:
#             r_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
#             count = count + 1
#             r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
#             r_eye = cv2.resize(r_eye, (24, 24))
#             r_eye = r_eye / 255
#             r_eye = r_eye.reshape(24, 24, -1)
#             r_eye = np.expand_dims(r_eye, axis=0)
#             rpred = model.predict(r_eye)
#             rpred = (rpred > 0.5).astype(int)
#             if rpred[0][0] == 1:
#                 lbl = 'Open'
#             else:
#                 lbl = 'Closed'
#             break

#         for (ex, ey, ew, eh) in left_eye:
#             l_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
#             count = count + 1
#             l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
#             l_eye = cv2.resize(l_eye, (24, 24))
#             l_eye = l_eye / 255
#             l_eye = l_eye.reshape(24, 24, -1)
#             l_eye = np.expand_dims(l_eye, axis=0)
#             lpred = model.predict(l_eye)
#             lpred = (lpred > 0.5).astype(int)
#             if lpred[0][0] == 1:
#                 lbl = 'Open'
#             else:
#                 lbl = 'Closed'
#             break

#     if rpred[0][0] == 0 and lpred[0][0] == 0:
#         score = score + 1
#         cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
#     else:
#         score = score - 1
#         cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#     if score < 0:
#         score = 0
#     cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#     if score > 15:
#         cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
#         try:
#             sound.play()
#         except:
#             pass
#         if thicc < 16:
#             thicc = thicc + 2
#         else:
#             thicc = thicc - 2
#             if thicc < 2:
#                 thicc = 2
#         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer

# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# lbl = 'Open'

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = [99]
# lpred = [99]

# # Initialize a flag to track alarm status
# alarm_on = False

# while True:
#     ret, frame = cap.read()
#     height, width = frame.shape[:2]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Detect faces in the frame
#     faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    
#     for (x, y, w, h) in faces:
#         # Draw a rectangle around the detected face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)
    
#         left_eye = leye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
#         right_eye = reye_cascade.detectMultiScale(gray[y:y+h, x:x+w])

#         for (ex, ey, ew, eh) in right_eye:
#             r_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
#             count = count + 1
#             r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
#             r_eye = cv2.resize(r_eye, (24, 24))
#             r_eye = r_eye / 255
#             r_eye = r_eye.reshape(24, 24, -1)
#             r_eye = np.expand_dims(r_eye, axis=0)
#             rpred = model.predict(r_eye)
#             rpred = (rpred > 0.5).astype(int)
#             if rpred[0][0] == 1:
#                 lbl = 'Open'
#             else:
#                 lbl = 'Closed'
#             break

#         for (ex, ey, ew, eh) in left_eye:
#             l_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
#             count = count + 1
#             l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
#             l_eye = cv2.resize(l_eye, (24, 24))
#             l_eye = l_eye / 255
#             l_eye = l_eye.reshape(24, 24, -1)
#             l_eye = np.expand_dims(l_eye, axis=0)
#             lpred = model.predict(l_eye)
#             lpred = (lpred > 0.5).astype(int)
#             if lpred[0][0] == 1:
#                 lbl = 'Open'
#             else:
#                 lbl = 'Closed'
#             break

#     if rpred[0][0] == 0 and lpred[0][0] == 0:
#         score = score + 1
#         cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
#     else:
#         score = score - 1
#         cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#     if score < 0:
#         score = 0

#     # Adjust the threshold here to trigger the alarm
#     if score < 5:
#         if not alarm_on:
#             try:
#                 sound.play()
#                 alarm_on = True
#             except:
#                 pass
#     else:
#         alarm_on = False

#     cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#     if score > 15:
#         cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
#         try:
#             sound.play()
#         except:
#             pass
#         if thicc < 16:
#             thicc = thicc + 2
#         else:
#             thicc = thicc - 2
#             if thicc < 2:
#                 thicc = 2
#         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer

# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# lbl = 'Open'

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = [99]
# lpred = [99]

# # Initialize a flag to track alarm status
# alarm_on = False

# while True:
#     ret, frame = cap.read()
#     height, width = frame.shape[:2]
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

#         left_eye = leye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
#         right_eye = reye_cascade.detectMultiScale(gray[y:y+h, x:x+w])

#         for (ex, ey, ew, eh) in right_eye:
#             r_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
#             count = count + 1
#             r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
#             r_eye = cv2.GaussianBlur(r_eye, (5, 5), 0)
#             r_eye = cv2.resize(r_eye, (24, 24))
#             r_eye = r_eye / 255
#             r_eye = r_eye.reshape(24, 24, -1)
#             r_eye = np.expand_dims(r_eye, axis=0)
#             rpred = model.predict(r_eye)
#             rpred = (rpred > 0.5).astype(int)

#         for (ex, ey, ew, eh) in left_eye:
#             l_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
#             count = count + 1
#             l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
#             l_eye = cv2.GaussianBlur(l_eye, (5, 5), 0)
#             l_eye = cv2.resize(l_eye, (24, 24))
#             l_eye = l_eye / 255
#             l_eye = l_eye.reshape(24, 24, -1)
#             l_eye = np.expand_dims(l_eye, axis=0)
#             lpred = model.predict(l_eye)
#             lpred = (lpred > 0.5).astype(int)

#     if rpred[0][0] == 0 and lpred[0][0] == 0:
#         score = score + 1
#         cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
#     else:
#         score = score - 1
#         cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#     if score < 0:
#         score = 0

#     # Adjust the threshold here to trigger the alarm
#     if score < 5:
#         if not alarm_on:
#             try:
#                 sound.play()
#                 alarm_on = True
#             except:
#                 pass
#     else:
#         alarm_on = False

#     cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#     if score > 15:
#         cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
#         try:
#             sound.play()
#         except:
#             pass
#         if thicc < 16:
#             thicc = thicc + 2
#         else:
#             thicc = thicc - 2
#             if thicc < 2:
#                 thicc = 2
#         cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     print(score)

# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# from flask import Flask, render_template, Response

# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# lbl = 'Open'

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = np.array([99])  # Initialize as NumPy array
# lpred = np.array([99])  # Initialize as NumPy array

# # Initialize a flag to track alarm status
# alarm_on = False

# app = Flask(__name__)

# def generate_frames():
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# @app.route('/')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(host='10.10.171.146', port=5000)




# import cv2
# import os
# from tensorflow.keras.models import load_model
# import numpy as np
# from pygame import mixer
# from flask import Flask, Response

# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# lbl = 'Open'

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = np.array([99])  # Initialize as NumPy array
# lpred = np.array([99])  # Initialize as NumPy array

# # Initialize a flag to track alarm status
# alarm_on = False

# app = Flask(__name__)

# def generate_frames():
#     global score, alarm_on
#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#             for (x, y, w, h) in faces:
#                 roi_gray = gray[y:y + h, x:x + w]
#                 roi_gray_resized = cv2.resize(roi_gray, (24, 24))

#                 img = np.array(roi_gray_resized) / 255.0
#                 img = np.reshape(img, (1, 24, 24, 1))

#                 # Predict the class
#                 pred = model.predict(img)
#                 # Get the predicted label
#                 if pred[0][0] > 0.5:
#                     lbl = 'Closed'
#                     score += 1  # Increment score if eyes are closed
#                 else:
#                     lbl = 'Open'
#                     score -= 1  # Decrease score if eyes are open

#                 # Ensure score doesn't go negative
#                 if score < 0:
#                     score = 0

#                 # Display the label and score on the frame
#                 cv2.putText(frame, f'Eyes: {lbl}', (x, y - 10), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                
#             # Display score continuously on every frame
#             cv2.putText(frame, f'Score: {score}', (10, frame.shape[0] - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     # app.run(host='172.20.10.4', port=5000)
#     app.run(host='127.0.0.1', port=5000)



# -------------
# ----------------------





import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from pygame import mixer

# Initialize the pygame mixer for sound alarm
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

lbl = 'Open'

# Attempt to load the pre-trained model
model = None
try:
    model = load_model('models/cnncat2.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

# Get the current working directory
path = os.getcwd()

# Open webcam feed
cap = cv2.VideoCapture(0)

# Font for displaying text on the video feed
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Initialize score and other parameters
count = 0
score = 0
thicc = 2
rpred = [99]  # Initialize as a list
lpred = [99]  # Initialize as a list

# Initialize a flag to track alarm status
alarm_on = False

# Main loop for face and eye detection
while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

        left_eye = leye_cascade.detectMultiScale(gray[y:y+h, x:x+w])
        right_eye = reye_cascade.detectMultiScale(gray[y:y+h, x:x+w])

        # Process right eye
        for (ex, ey, ew, eh) in right_eye:
            r_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
            count = count + 1
            r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
            r_eye = cv2.GaussianBlur(r_eye, (5, 5), 0)
            r_eye = cv2.resize(r_eye, (24, 24))
            r_eye = r_eye / 255
            r_eye = r_eye.reshape(24, 24, -1)
            r_eye = np.expand_dims(r_eye, axis=0)

            # Only make a prediction if the model is loaded
            if model is not None:
                rpred = model.predict(r_eye)
                rpred = (rpred > 0.5).astype(int)

        # Process left eye
        for (ex, ey, ew, eh) in left_eye:
            l_eye = frame[y+ey:y+ey+eh, x+ex:x+ex+ew]
            count = count + 1
            l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
            l_eye = cv2.GaussianBlur(l_eye, (5, 5), 0)
            l_eye = cv2.resize(l_eye, (24, 24))
            l_eye = l_eye / 255
            l_eye = l_eye.reshape(24, 24, -1)
            l_eye = np.expand_dims(l_eye, axis=0)

            # Only make a prediction if the model is loaded
            if model is not None:
                lpred = model.predict(l_eye)
                lpred = (lpred > 0.5).astype(int)

    # Ensure that rpred and lpred are arrays and have the correct shape
    if isinstance(rpred, np.ndarray) and len(rpred.shape) == 2:
        rpred = rpred[0]
    if isinstance(lpred, np.ndarray) and len(lpred.shape) == 2:
        lpred = lpred[0]

    if rpred[0] == 0 and lpred[0] == 0:
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score < 0:
        score = 0

    # Adjust the threshold here to trigger the alarm
    if score < 5:
        if not alarm_on:
            try:
                sound.play()
                alarm_on = True
            except:
                pass
    else:
        alarm_on = False

    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass
        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(score)

cap.release()
cv2.destroyAllWindows()



























# --------------

# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# from flask import Flask, Response
# import dlib

# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# lbl = 'Open'

# model = load_model('models/cnncat2.h5')
# path = os.getcwd()
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = np.array([99])  # Initialize as NumPy array
# lpred = np.array([99])  # Initialize as NumPy array

# # Initialize a flag to track alarm status
# alarm_on = False

# app = Flask(__name__)

# # Initialize the dlib correlation tracker
# tracker = dlib.correlation_tracker()

# # Initialize variables for tracking
# tracking = False
# bbox = None

# def generate_frames():
#     global tracking, bbox

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             if not tracking:
#                 # Detect faces
#                 faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
#                 for (x, y, w, h) in faces:
#                     # Initialize the tracker with the bounding box of the detected face
#                     bbox = (x, y, x + w, y + h)
#                     tracker.start_track(frame, dlib.rectangle(*bbox))
#                     tracking = True
#                     break  # Only consider the first detected face

#             else:
#                 # Update the tracker and get the new bounding box
#                 tracker.update(frame)
#                 bbox = (int(tracker.get_position().left()),
#                         int(tracker.get_position().top()),
#                         int(tracker.get_position().width()),
#                         int(tracker.get_position().height()))

#                 # Check if the bounding box is not empty
#                 if bbox[1] is not None:
#                     bbox = cv2.boundingRect(bbox[1])
#                     x, y, w, h = bbox
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                     # Extract the region of interest (face) for further processing
#                     roi_gray = gray[y:y + h, x:x + w]
#                     roi_gray_resized = cv2.resize(roi_gray, (24, 24))

#                     img = np.array(roi_gray_resized) / 255.0
#                     img = np.reshape(img, (1, 24, 24, 1))

#                     # Predict the class
#                     pred = model.predict(img)
#                     # Get the predicted label
#                     if pred[0][0] > 0.5:
#                         lbl = 'Closed'
#                     else:
#                         lbl = 'Open'
#                     # Display the label on the frame
#                     cv2.putText(frame, f'Eyes: {lbl}', (x, y - 10), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=5000)





















# import cv2
# import os
# from keras.models import load_model
# import numpy as np
# from pygame import mixer
# from flask import Flask, Response
# import dlib

# mixer.init()
# sound = mixer.Sound('alarm.wav')

# face_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
# leye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
# reye_cascade = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# lbl = 'Open'

# # Ensure model is loaded correctly
# try:
#     model = load_model('models/cnncat2.h5')
# except Exception as e:
#     print(f"Error loading model: {e}")

# path = os.getcwd()
# cap = cv2.VideoCapture(0)
# font = cv2.FONT_HERSHEY_COMPLEX_SMALL
# count = 0
# score = 0
# thicc = 2
# rpred = np.array([99])
# lpred = np.array([99])

# alarm_on = False

# app = Flask(__name__)

# tracker = dlib.correlation_tracker()

# tracking = False
# bbox = None

# def generate_frames():
#     global tracking, bbox, alarm_on

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break
#         else:
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             if not tracking:
#                 faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#                 for (x, y, w, h) in faces:
#                     bbox = (x, y, x + w, y + h)
#                     tracker.start_track(frame, dlib.rectangle(*bbox))
#                     tracking = True
#                     break

#             else:
#                 tracker.update(frame)
#                 bbox = (int(tracker.get_position().left()),
#                         int(tracker.get_position().top()),
#                         int(tracker.get_position().width()),
#                         int(tracker.get_position().height()))

#                 if bbox:
#                     x, y, w, h = bbox
#                     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#                     roi_gray = gray[y:y + h, x:x + w]
#                     roi_gray_resized = cv2.resize(roi_gray, (24, 24))

#                     img = np.array(roi_gray_resized) / 255.0
#                     img = np.reshape(img, (1, 24, 24, 1))

#                     pred = model.predict(img)
#                     if pred[0][0] > 0.5:
#                         lbl = 'Closed'
#                     else:
#                         lbl = 'Open'
#                     cv2.putText(frame, f'Eyes: {lbl}', (x, y - 10), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

#                     if lbl == 'Closed' and not alarm_on:
#                         sound.play()
#                         alarm_on = True
#                     elif lbl == 'Open' and alarm_on:
#                         sound.stop()
#                         alarm_on = False

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=5000)
