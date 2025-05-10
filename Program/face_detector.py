import cv2
import numpy as np

class FaceDetector:
    def __init__(self):
        # Arcfelismerő betöltése (OpenCV beépített Haar cascade)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
    def detect_face(self, frame):
        # Szürkeárnyalatossá konvertálás
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Arcok detektálása - paraméterek finomítása a jobb felismerésért
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(120, 120))
        
        face_landmarks = []
        
        for (x, y, w, h) in faces:
            # Szemek detektálása az arcon belül
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 4, minSize=(30, 30))
            
            # Egyszerűsített arcjellemzők létrehozása
            # Szemek
            eye_points = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Maximum 2 szem
                eye_center = (x + ex + ew//2, y + ey + eh//2)
                eye_points.append(eye_center)
            
            # Ha nem talált szemeket, becsüljük őket
            if len(eye_points) == 0:
                eye_y = y + int(h * 0.35)  # Szem magassága az arcon
                eye_points = [(x + w//3, eye_y), (x + 2*w//3, eye_y)]
            elif len(eye_points) == 1:
                eye_x, eye_y = eye_points[0]
                if eye_x < x + w//2:  # bal szem
                    eye_points.append((x + 2*w//3, eye_y))
                else:  # jobb szem
                    eye_points.insert(0, (x + w//3, eye_y))
            
            # Arckerethez szorosan illeszkedő téglalap
            face_rect = (x, y, x+w, y+h)
            
            # Száj pozíciójának becslése (az arc alsó harmada)
            mouth_y = y + int(h * 0.75)
            mouth_points = [(x + w//4, mouth_y), (x + w//2, mouth_y), (x + 3*w//4, mouth_y)]
            
            # Orr pozíciójának becslése (az arc közepe)
            nose_x, nose_y = x + w//2, y + int(h * 0.55)
            nose_points = [(nose_x, nose_y)]
            
            # Fejtetőpont becslése - lejjebb pozicionálva
            forehead_y = y + int(h * 0.1)  # Homlok magassága
            top_head_points = [(x + w//4, forehead_y), (x + w//2, y), (x + 3*w//4, forehead_y)]
            
            # Összes pont összegyűjtése
            all_points = eye_points + mouth_points + nose_points + top_head_points
            
            face_landmarks.append({
                'rect': face_rect,
                'points': all_points,
                'eyes': eye_points,
                'mouth': mouth_points,
                'nose': nose_points,
                'top_head': top_head_points
            })
        
        return face_landmarks
    
    def get_face_regions(self, landmarks):
        """Visszaadja a különböző arcrégiók koordinátáit"""
        if not landmarks:
            return None
        
        face_data = landmarks[0]  # Csak az első arcot használjuk
        
        # Bal és jobb szem szétválasztása
        eyes = face_data.get('eyes', [])
        left_eye = []
        right_eye = []
        
        # Ha legalább két szem található, elkülönítjük őket x koordináta alapján
        if len(eyes) >= 2:
            sorted_eyes = sorted(eyes, key=lambda p: p[0])
            left_eye = [sorted_eyes[0]]
            right_eye = [sorted_eyes[1]]
        elif len(eyes) == 1:
            # Ha csak egy szem található, megbecsüljük a másikat
            eye_x, eye_y = eyes[0]
            face_rect = face_data['rect']
            face_center_x = (face_rect[0] + face_rect[2]) // 2
            
            if eye_x < face_center_x:
                left_eye = [eyes[0]]
                # Jobb szem becslése
                right_eye = [(face_rect[2] - (eye_x - face_rect[0]), eye_y)]
            else:
                right_eye = [eyes[0]]
                # Bal szem becslése
                left_eye = [(face_rect[0] + (face_rect[2] - eye_x)), eye_y]
        
        return {
            'left_eye': left_eye,
            'right_eye': right_eye,
            'top_head': face_data.get('top_head', []),
            'nose': face_data.get('nose', []),
            'mouth': face_data.get('mouth', []),
            'face_rect': face_data.get('rect', None)
        }