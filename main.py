from ultralytics import YOLO
import cv2
from module.my_pca import MyPCA
# from module.my_svd import MySVD
from module.utils import load_images
import numpy as np
import torch

X_train , y_train = load_images('data/train' , img_size=(80,80))
model = MyPCA(n_components=50)
X_train_project = model.fit.transform(X_train)

def recognize_face(face_vector, model, db_features, db_labels, threshold=9000):
    new_project = model.transform(face_vector.reshape(1,-1))[0]
    dists = np.linalg.norm(db_features - new_project , axis=1)
    idx = np.argmin(dists)
    return db_labels[idx] if dists[idx] < threshold else "Unknown"

yolo = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = yolo(frame, device = 'cuda')
    for box in results[0].boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0]).cpu().numpy()
        face= frame[y1:y2, x1:x2]
        if face.size == 0:
            continue
        try:
            face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face_resized = cv2.resize(face_gray, (80,80)).flatten()
        except Exception:
            continue
        name = recognize(face_resized, model, X_train_project, y_train)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame,name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) == 27: break
    cap.release()
    cv2.destroyAllWindows()

