import cv2
import numpy as np
import os
import time
from face_detector import FaceDetector

class VirtualAccessoryApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.face_detector = FaceDetector()
        
        # Kamera felbontás beállítása nagyobbra
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Kiegészítő kategóriák és elemek
        self.categories = {
            "fejfedok": ["hat", "cap", "crown", "santa_hat"],
            "szemuvegek": ["sunglasses", "nerd_glasses", "party_glasses"],
            "arcszorszet": ["mustache1", "beard", "goatee"]
        }
        
        # Aktív kategória és kiegészítők
        self.active_category = "fejfedok"
        self.active_accessories = {}
        
        # Testreszabási beállítások
        self.customization = {
            "size_factor": {},
            "position_offset": {}
        }
        
        # Aktuális képhatás
        self.available_filters = ["none", "grayscale", "sepia", "cartoon", "negative"]
        self.current_filter = "none"
        
        # Debug mód
        self.debug_mode = False
        
        # Kiegészítők betöltése
        self.accessories = self.load_accessories()
        
        # Kiválasztott kiegészítő
        self.selected_accessory = None
        
        # Kategória elemek indexei a kiválasztáshoz
        self.category_selected_index = {"fejfedok": 0, "szemuvegek": 0, "arcszorszet": 0}
        
        # Alapbeállítások minden kiegészítőhöz
        for acc in self.accessories:
            self.customization["size_factor"][acc] = 1.0
            self.customization["position_offset"][acc] = (0, 0)
        
        # Alapértelmezett kiegészítő
        if "hat" in self.accessories:
            self.active_accessories["hat"] = True
            self.selected_accessory = "hat"
    
    def load_accessories(self):
        """Kiegészítők betöltése"""
        accessories = {}
        
        print("Kiegészítők betöltése...")
        
        # Accessories mappából minden png betöltése
        if os.path.exists("accessories"):
            png_files = [f for f in os.listdir("accessories") if f.endswith(".png")]
            for png_file in png_files:
                name = os.path.splitext(png_file)[0]
                path = os.path.join("accessories", png_file)
                image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if image is not None:
                    print(f"  {name} betöltve: {path}")
                    accessories[name] = image
        
        # Alapkiegészítők generálása, ha hiányoznak
        if len(accessories) == 0:
            print("Alapkiegészítők generálása...")
            
            # Kalap
            hat = np.zeros((150, 200, 4), dtype=np.uint8)
            cv2.ellipse(hat, (100, 100), (80, 40), 0, 0, 360, (42, 42, 165, 255), -1)
            cv2.ellipse(hat, (100, 70), (50, 40), 0, 0, 180, (42, 85, 165, 255), -1)
            accessories["hat"] = hat
            
            # Napszemüveg
            sunglasses = np.zeros((100, 200, 4), dtype=np.uint8)
            cv2.rectangle(sunglasses, (10, 10), (90, 40), (0, 0, 0, 255), -1)
            cv2.rectangle(sunglasses, (110, 10), (190, 40), (0, 0, 0, 255), -1)
            accessories["sunglasses"] = sunglasses
            
            # Bajusz
            mustache = np.zeros((60, 120, 4), dtype=np.uint8)
            cv2.ellipse(mustache, (60, 30), (50, 20), 0, 0, 180, (0, 0, 0, 255), -1)
            accessories["mustache1"] = mustache
            
        return accessories
    
    def process_frame(self, frame):
        """Képkocka feldolgozása"""
        # Arc detektálása
        face_landmarks = self.face_detector.detect_face(frame)
        face_regions = self.face_detector.get_face_regions(face_landmarks)
        
        # Képhatás alkalmazása
        processed_frame = self.apply_filter(frame.copy(), self.current_filter)
        
        # Kiegészítők ráhelyezése
        if face_regions:
            for accessory_name, is_active in self.active_accessories.items():
                if is_active and accessory_name in self.accessories:
                    processed_frame = self.overlay_accessory(processed_frame, accessory_name, face_regions)
        
        # Debug információk
        if self.debug_mode and face_landmarks:
            for face in face_landmarks:
                x, y, x2, y2 = face['rect']
                cv2.rectangle(processed_frame, (x, y), (x2, y2), (0, 255, 0), 2)
        
        # UI elemek hozzáadása
        processed_frame = self.add_ui_elements(processed_frame)
        
        return processed_frame, face_landmarks
    
    def add_ui_elements(self, frame):
        """UI elemek hozzáadása a képhez"""
        # Felső sáv
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (50, 50, 50), -1)
        
        # Kategória gombok
        button_width = 120
        x_pos = 10
        for category in self.categories:
            if category == self.active_category:
                color = (0, 200, 200)
            else:
                color = (200, 200, 0)
            cv2.rectangle(frame, (x_pos, 5), (x_pos + button_width, 35), color, -1)
            cv2.putText(frame, category.capitalize(), (x_pos + 10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            x_pos += button_width + 10
        
        # Szűrő kijelzése
        filter_text = f"Szűrő: {self.current_filter}"
        cv2.putText(frame, filter_text, (frame.shape[1] - 200, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Aktív kiegészítők kijelzése
        active_text = "Aktív: " + ", ".join([acc for acc, active in self.active_accessories.items() if active])
        cv2.putText(frame, active_text, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Kategória kiegészítők listája
        y_pos = 50
        for i, item in enumerate(self.categories[self.active_category]):
            if item in self.accessories:
                # Kiválasztott elem megjelenítése másképp
                is_selected = (i == self.category_selected_index[self.active_category])
                is_active = self.active_accessories.get(item, False)
                
                if is_active:
                    color = (0, 200, 0)
                elif is_selected:
                    color = (0, 100, 200)
                else:
                    color = (150, 150, 150)
                
                cv2.rectangle(frame, (10, y_pos), (150, y_pos + 30), color, -1)
                cv2.putText(frame, item, (20, y_pos + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_pos += 40
        
        # Vezérlők
        if self.selected_accessory:
            ctrl_x = frame.shape[1] - 200
            ctrl_y = 60
            
            # Méretezés vezérlők
            cv2.putText(frame, f"Méret: {self.selected_accessory}", (ctrl_x, ctrl_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.rectangle(frame, (ctrl_x, ctrl_y + 10), (ctrl_x + 30, ctrl_y + 30), (0, 100, 200), -1)
            cv2.putText(frame, "-", (ctrl_x + 10, ctrl_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.rectangle(frame, (ctrl_x + 40, ctrl_y + 10), (ctrl_x + 70, ctrl_y + 30), (0, 200, 100), -1)
            cv2.putText(frame, "+", (ctrl_x + 50, ctrl_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Pozícionálás vezérlők
            ctrl_y += 40
            cv2.putText(frame, "Pozíció:", (ctrl_x, ctrl_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Nyilak
            arrow_y = ctrl_y + 20
            # Fel
            cv2.rectangle(frame, (ctrl_x + 35, arrow_y), (ctrl_x + 65, arrow_y + 30), (150, 150, 200), -1)
            cv2.putText(frame, "↑", (ctrl_x + 45, arrow_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Bal
            arrow_y += 35
            cv2.rectangle(frame, (ctrl_x, arrow_y), (ctrl_x + 30, arrow_y + 30), (150, 150, 200), -1)
            cv2.putText(frame, "←", (ctrl_x + 10, arrow_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Le
            cv2.rectangle(frame, (ctrl_x + 35, arrow_y), (ctrl_x + 65, arrow_y + 30), (150, 150, 200), -1)
            cv2.putText(frame, "↓", (ctrl_x + 45, arrow_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            # Jobb
            cv2.rectangle(frame, (ctrl_x + 70, arrow_y), (ctrl_x + 100, arrow_y + 30), (150, 150, 200), -1)
            cv2.putText(frame, "→", (ctrl_x + 80, arrow_y + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Billentyűzet-kiosztás segítség
        self.add_keyboard_help(frame)
        
        return frame
    
    def add_keyboard_help(self, frame):
        """Billentyűzet-kiosztás segítség megjelenítése"""
        # Háttér a jobb oldalon
        help_width = 220
        help_start_x = frame.shape[1] - help_width - 10
        help_start_y = 250
        help_end_y = help_start_y + 340
        
        cv2.rectangle(frame, (help_start_x, help_start_y), 
                     (frame.shape[1] - 10, help_end_y), 
                     (50, 50, 50, 150), -1)
        
        # Cím
        cv2.putText(frame, "Billentyű segítség:", 
                   (help_start_x + 10, help_start_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Billentyű-parancsok
        commands = [
            "1,2,3: Kategória váltás",
            "Fel/Le: Elem kiválasztás",
            "Space: Elem be/kikapcsolás",
            "5: Szűrő váltás",
            "+/-: Méretezés",
            "W,A,S,D: Mozgatás",
            "P: Kép mentése",
            "V: Debug mód",
            "ESC: Kilépés"
        ]
        
        for i, cmd in enumerate(commands):
            cv2.putText(frame, cmd, 
                       (help_start_x + 15, help_start_y + 55 + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        return frame
    
    def apply_filter(self, frame, filter_name):
        """Képhatás alkalmazása"""
        if filter_name == "none":
            return frame
        
        elif filter_name == "grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        elif filter_name == "sepia":
            kernel = np.array([
                [0.272, 0.534, 0.131],
                [0.349, 0.686, 0.168],
                [0.393, 0.769, 0.189]
            ])
            sepia = cv2.transform(frame, kernel)
            return np.clip(sepia, 0, 255).astype(np.uint8)
        
        elif filter_name == "negative":
            return 255 - frame
        
        elif filter_name == "cartoon":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                         cv2.THRESH_BINARY, 9, 9)
            color = cv2.bilateralFilter(frame, 9, 300, 300)
            cartoon = cv2.bitwise_and(color, color, mask=edges)
            return cartoon
        
        return frame
    
    def overlay_accessory(self, frame, accessory_name, face_regions):
        """Kiegészítő ráhelyezése a képre"""
        if accessory_name not in self.accessories or not face_regions:
            return frame
        
        accessory = self.accessories[accessory_name]
        
        # Arcadatok kinyerése
        face_rect = face_regions.get('face_rect')
        if not face_rect:
            return frame
            
        x, y, x2, y2 = face_rect
        face_width = x2 - x
        face_height = y2 - y
        
        # Testreszabási paraméterek kinyerése
        size_factor = self.customization["size_factor"].get(accessory_name, 1.0)
        position_offset = self.customization["position_offset"].get(accessory_name, (0, 0))
        
        # Kiegészítő elhelyezése a kategória szerint
        if accessory_name in ["hat", "crown", "santa_hat", "cap"]:
            # Fejfedők
            base_scale = face_width / accessory.shape[1] * 1.2 * size_factor
            width = int(accessory.shape[1] * base_scale)
            height = int(accessory.shape[0] * base_scale)
            
            pos_x = x + (face_width - width) // 2 + position_offset[0]
            pos_y = max(0, y - height // 2) + position_offset[1]
            
            self.place_accessory(frame, accessory, pos_x, pos_y, width, height)
            
        elif accessory_name in ["sunglasses", "nerd_glasses", "party_glasses"]:
            # Szemüvegek
            if not face_regions.get('left_eye') or not face_regions.get('right_eye'):
                return frame
                
            left_eye = face_regions['left_eye'][0]
            right_eye = face_regions['right_eye'][0]
            
            eye_distance = np.sqrt((right_eye[0] - left_eye[0])**2 + (right_eye[1] - left_eye[1])**2)
            
            base_scale = eye_distance / (accessory.shape[1] * 0.4) * size_factor
            width = int(accessory.shape[1] * base_scale)
            height = int(accessory.shape[0] * base_scale)
            
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            
            pos_x = center_x - width // 2 + position_offset[0]
            pos_y = center_y - height // 2 + position_offset[1]
            
            self.place_accessory(frame, accessory, pos_x, pos_y, width, height)
            
        elif accessory_name in ["mustache1", "beard", "goatee"]:
            # Arcszőrzet
            if not face_regions.get('nose') or not face_regions.get('mouth'):
                return frame
                
            nose = face_regions['nose'][0]
            mouth = face_regions['mouth'][1] if len(face_regions['mouth']) > 1 else face_regions['mouth'][0]
            
            base_scale = face_width / (accessory.shape[1] * 2) * size_factor
            width = int(accessory.shape[1] * base_scale)
            height = int(accessory.shape[0] * base_scale)
            
            # Pozicionálás
            if accessory_name == "mustache1":
                pos_y = int((nose[1] + mouth[1]) / 2) - height // 2
            else:
                pos_y = mouth[1] - height // 4
            
            pos_x = x + (face_width - width) // 2 + position_offset[0]
            pos_y += position_offset[1]
            
            self.place_accessory(frame, accessory, pos_x, pos_y, width, height)
        
        return frame
    
    def place_accessory(self, frame, accessory, x, y, width, height):
        """Kiegészítő ráhelyezése a képre"""
        try:
            # Átméretezés
            resized = cv2.resize(accessory, (width, height))
            
            # Biztosítsuk, hogy a pozíciók a képen belül vannak
            if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
                # Kiszámítjuk a látható részt
                x_visible_start = max(0, x)
                y_visible_start = max(0, y)
                x_visible_end = min(frame.shape[1], x + width)
                y_visible_end = min(frame.shape[0], y + height)
                
                # Kiszámítjuk a megfelelő indexeket az átméretezett kiegészítőben
                x_offset_in_accessory = x_visible_start - x
                y_offset_in_accessory = y_visible_start - y
                width_to_use = x_visible_end - x_visible_start
                height_to_use = y_visible_end - y_visible_start
                
                if width_to_use <= 0 or height_to_use <= 0:
                    return  # Nincs látható rész
                
                # Kivágjuk csak a látható részt az átméretezett kiegészítőből
                cropped_accessory = resized[
                    y_offset_in_accessory:y_offset_in_accessory + height_to_use,
                    x_offset_in_accessory:x_offset_in_accessory + width_to_use
                ]
                
                # Kivágjuk a megfelelő részt a képből
                roi = frame[
                    y_visible_start:y_visible_end,
                    x_visible_start:x_visible_end
                ]
                
                # Alkalmazzuk az átlátszóságot
                if cropped_accessory.shape[2] == 4:
                    alpha = cropped_accessory[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    alpha = np.repeat(alpha, 3, axis=2)
                    result = (1.0 - alpha) * roi + alpha * cropped_accessory[:, :, :3]
                    frame[y_visible_start:y_visible_end, x_visible_start:x_visible_end] = result.astype(np.uint8)
            else:
                # A kiegészítő teljesen a képen belül van
                roi = frame[y:y+height, x:x+width]
                
                # Alkalmazzuk az átlátszóságot
                if resized.shape[2] == 4:
                    alpha = resized[:, :, 3] / 255.0
                    alpha = np.expand_dims(alpha, axis=2)
                    alpha = np.repeat(alpha, 3, axis=2)
                    result = (1.0 - alpha) * roi + alpha * resized[:, :, :3]
                    frame[y:y+height, x:x+width] = result.astype(np.uint8)
        
        except Exception as e:
            print(f"Hiba a kiegészítő elhelyezése során: {e}")
    
    def select_current_accessory(self):
        """Kiválasztja és aktiválja/deaktiválja az aktuálisan kiválasztott kiegészítőt"""
        if not self.categories[self.active_category]:
            return
            
        index = self.category_selected_index[self.active_category]
        if index < len(self.categories[self.active_category]):
            item = self.categories[self.active_category][index]
            if item in self.accessories:
                self.active_accessories[item] = not self.active_accessories.get(item, False)
                if self.active_accessories[item]:
                    self.selected_accessory = item
                else:
                    # Ha kikapcsoltuk, akkor keressünk egy másik aktív kiegészítőt
                    self.selected_accessory = next(
                        (acc for acc, active in self.active_accessories.items() if active), 
                        None
                    )
    
    def run(self):
        """Alkalmazás futtatása"""
        print("Kamera inicializálása...")
        if not self.cap.isOpened():
            print("HIBA: A kamera nem nyitható meg!")
            return
        
        # Ablak létrehozása és átméretezése
        cv2.namedWindow("Virtualis Probara Probalo Pro", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Virtualis Probara Probalo Pro", 1280, 720)
        
        while True:
            # Kamera kép olvasása
            ret, frame = self.cap.read()
            if not ret:
                print("HIBA: Nem sikerült képkockát olvasni a kamerából!")
                break
            
            # Tükrözés
            frame = cv2.flip(frame, 1)
            
            # Kép feldolgozása közvetlenül itt, threading nélkül
            processed_frame, _ = self.process_frame(frame)
            
            # Feldolgozott kép megjelenítése
            cv2.imshow("Virtualis Probara Probalo Pro", processed_frame)
            
            # Billentyű kezelése
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
            elif key == ord('1'):
                # Kategória váltás fejfedőkre
                self.active_category = "fejfedok"
            elif key == ord('2'):
                # Kategória váltás szemüvegekre
                self.active_category = "szemuvegek"
            elif key == ord('3'):
                # Kategória váltás arcszőrzetre
                self.active_category = "arcszorszet"
            elif key == ord(' '):
                # Aktuálisan kiválasztott elem be/kikapcsolása
                self.select_current_accessory()
            elif key == ord('5'):
                # Szűrő váltás
                idx = (self.available_filters.index(self.current_filter) + 1) % len(self.available_filters)
                self.current_filter = self.available_filters[idx]
            elif key == ord('+') or key == ord('='):
                # Méret növelése
                if self.selected_accessory:
                    current_size = self.customization["size_factor"].get(self.selected_accessory, 1.0)
                    self.customization["size_factor"][self.selected_accessory] = min(2.0, current_size + 0.1)
            elif key == ord('-'):
                # Méret csökkentése
                if self.selected_accessory:
                    current_size = self.customization["size_factor"].get(self.selected_accessory, 1.0)
                    self.customization["size_factor"][self.selected_accessory] = max(0.5, current_size - 0.1)
            elif key == ord('w'):
                # Fel mozgatás
                if self.selected_accessory:
                    current_offset = self.customization["position_offset"].get(self.selected_accessory, (0, 0))
                    self.customization["position_offset"][self.selected_accessory] = (
                        current_offset[0], current_offset[1] - 5
                    )
            elif key == ord('s'):
                # Le mozgatás
                if self.selected_accessory:
                    current_offset = self.customization["position_offset"].get(self.selected_accessory, (0, 0))
                    self.customization["position_offset"][self.selected_accessory] = (
                        current_offset[0], current_offset[1] + 5
                    )
            elif key == ord('a'):
                # Balra mozgatás
                if self.selected_accessory:
                    current_offset = self.customization["position_offset"].get(self.selected_accessory, (0, 0))
                    self.customization["position_offset"][self.selected_accessory] = (
                        current_offset[0] - 5, current_offset[1]
                    )
            elif key == ord('d'):
                # Jobbra mozgatás
                if self.selected_accessory:
                    current_offset = self.customization["position_offset"].get(self.selected_accessory, (0, 0))
                    self.customization["position_offset"][self.selected_accessory] = (
                        current_offset[0] + 5, current_offset[1]
                    )
            elif key == ord('v'):
                # Debug mód be/kikapcsolása
                self.debug_mode = not self.debug_mode
            elif key == ord('p'):
                # Kép mentése
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"selfie_{timestamp}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Kép mentve: {filename}")
            elif key == 82:  # Fel nyíl
                # Előző elem kiválasztása
                index = self.category_selected_index[self.active_category]
                if index > 0:
                    self.category_selected_index[self.active_category] = index - 1
                    item = self.categories[self.active_category][self.category_selected_index[self.active_category]]
                    if item in self.accessories and self.active_accessories.get(item, False):
                        self.selected_accessory = item
            elif key == 84:  # Le nyíl
                # Következő elem kiválasztása
                index = self.category_selected_index[self.active_category]
                if index < len(self.categories[self.active_category]) - 1:
                    self.category_selected_index[self.active_category] = index + 1
                    item = self.categories[self.active_category][self.category_selected_index[self.active_category]]
                    if item in self.accessories and self.active_accessories.get(item, False):
                        self.selected_accessory = item
        
        # Kamera és erőforrások felszabadítása
        self.cap.release()
        cv2.destroyAllWindows()
        print("Alkalmazás leállítva.")

if __name__ == "__main__":
    app = VirtualAccessoryApp()
    app.run()