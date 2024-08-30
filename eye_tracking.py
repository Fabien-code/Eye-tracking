import time
import cv2
import numpy as np
from mouse_tests import Screen_Mouse

def initialize_cascades():
    """Initialiser les classificateurs en cascade pour les visages et les yeux."""
    face_cascade = cv2.CascadeClassifier("C:/Users/FireF/OneDrive/Documents/GitHub/Eye-tracking/cascades/haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("C:/Users/FireF/OneDrive/Documents/GitHub/Eye-tracking/cascades/haarcascade_eye.xml")
    return face_cascade, eye_cascade

def capture_frame(cap):
    """Capturer une image depuis la webcam."""
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def detect_faces(gray_frame, face_cascade):
    """Détecter les visages dans l'image en niveaux de gris."""
    return face_cascade.detectMultiScale(gray_frame, 1.3, 5)


def quantize_grayscale(image, num_levels):
    # Normaliser l'image en niveaux de gris pour obtenir une image à 8 bits
    image_normalized = image.astype(np.float32) / 255.0

    # Quantifier les niveaux de gris
    quantized_image = (image_normalized * (num_levels - 1)).astype(np.uint8)  # Quantification en niveaux
    quantized_image = (quantized_image * (255.0 / (num_levels - 1))).astype(np.uint8)  # Échelle de 0 à 255

    return quantized_image

def crop_eye(image, scale_factor=100):
    """Rogner l'image en appliquant un facteur de réduction."""
    if image is not None:
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        center_x, center_y = width // 2, height // 2
        x1 = max(center_x - new_width // 2, 0)
        y1 = max(center_y - new_height // 2, 0)
        x2 = min(center_x + new_width // 2, width)
        y2 = min(center_y + new_height // 2, height)

        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    return None

def detect_eyes(roi_gray, eye_cascade):
    """Détecter les yeux dans la région du visage en niveaux de gris."""
    return eye_cascade.detectMultiScale(roi_gray)

def extract_right_eye(roi_color, eyes):
    """Extraire l'œil droit de la région du visage."""
    if len(eyes) >= 2:
        # On suppose que les yeux sont détectés dans l'ordre de gauche à droite
        eye1, eye2 = sorted(eyes, key=lambda e: e[0])[:2]
        ex, ey, ew, eh = eye2  # Choisir l'œil droit (deuxième en partant de la gauche)
        return roi_color[ey:ey + eh, ex:ex + ew]
    return None

def apply_mask(image):
    """Appliquer un masque pour gommer les lignes horizontales noires foncées dans l'image en niveaux de gris."""
    if image is not None:
        
        # Seuiller pour détecter les lignes noires (foncées)
        _, mask = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)
        
        # Inverser le masque pour que les lignes noires deviennent blanches
        

        # Appliquer le masque à l'image
        masked_image = cv2.bitwise_and(image, image, mask=mask)

        return masked_image
    return None


def find_darkest_point(image):
    """
    Trouver les coordonnées du point le plus noir dans une image en niveaux de gris.
    
    Args:
    - image (numpy.ndarray): L'image en niveaux de gris où l'on cherche le point le plus noir.

    Returns:
    - tuple: Les coordonnées (x, y) du point le plus noir.
    """
    if image is None:
        return None
    
    # Trouver la valeur minimale et ses indices
    min_val, _, min_loc, _ = cv2.minMaxLoc(image)
    
    return min_loc

def normalize_pair(size_image, size_screen, pair):
    x, y = pair
    x = int(x*size_screen[0]/size_image[0])
    y = int(y*size_screen[1]/size_image[1])
    return x,y

def main():
    mouse  = Screen_Mouse()
    # Initialiser les classificateurs en cascade
    face_cascade, eye_cascade = initialize_cascades()

    # Démarrer la capture vidéo depuis la webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Right Eye', cv2.WINDOW_NORMAL)

    while True:
        time.sleep(0.05)  # Petite pause pour éviter une utilisation excessive du CPU
        # Capturer une image
        frame = capture_frame(cap)
        if frame is None:
            break

        # Convertir l'image en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_levels = 50
        gray = quantize_grayscale(gray, num_levels)
        # Détecter les visages
        faces = detect_faces(gray, face_cascade)
        for (x, y, w, h) in faces:
            # Extraire la région du visage
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Détecter les yeux
            eyes = detect_eyes(roi_gray, eye_cascade)

            # Extraire et afficher l'œil droit
            right_eye = extract_right_eye(roi_gray, eyes)
            
            if right_eye is not None:
                # Rogner l'œil droit pour le rendre plus petit
                cropped_eye = crop_eye(right_eye, scale_factor=0.6)
                if cropped_eye is not None:
                    masked_eye = apply_mask(cropped_eye)
                    if masked_eye is not None:
                        cv2.imshow('Right Eye', masked_eye)
                        darkest_point = find_darkest_point(masked_eye)
                        size_image = masked_eye.shape
                        size_screen = mouse.get_screen_size()
                        #mouse.drag_to(normalize_pair(size_image,size_screen,darkest_point)[0],normalize_pair(size_image,size_screen,darkest_point)[1],0.05)
                        
        # Quitter avec la touche 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
