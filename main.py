import cv2
from ultralytics import YOLO
import easyocr

# Initialize models
model = YOLO("models/license_plate_best.pt")
reader = easyocr.Reader(['en'], gpu=True)

def correct_plate_format(ocr_text):
    ocr_text = ocr_text.upper().replace(" ", "")
    return "".join(ch for ch in ocr_text if ch.isalnum())

def recognize_plate(plate_crop):
    if plate_crop.size == 0:
        return ""

    # Upscale for EasyOCR
    plate_resized = cv2.resize(plate_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    try:
        ocr_result = reader.readtext(
            plate_resized, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', paragraph=True
        )
        if len(ocr_result) > 0:
            best_candidate = ""
            for text in ocr_result:
                candidate = correct_plate_format(text)
                if len(candidate) >= 4:
                    has_digit = any(c.isdigit() for c in candidate)
                    if has_digit and len(candidate) == 7:
                        return candidate 
                    
                    if has_digit and len(candidate) > len(best_candidate):
                        best_candidate = candidate
                    elif not best_candidate and len(candidate) > len(best_candidate):
                        best_candidate = candidate
            return best_candidate
    except Exception:
        pass

    return ""

# ---- CONFIGURATION ----
input_image_path = "vehicle_image.jpeg"       
output_image_path = "output_annotated.jpg"   
output_text_path = "extracted_plates.txt"    

frame = cv2.imread(input_image_path)
extracted_plates_list = []

if frame is None:
    print(f"❌ Error: Could not find or load image at '{input_image_path}'")
else:
    CONF_THRESH = 0.25  
    results = model(frame, verbose=False)
    active_overlays = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.cpu().numpy()[0])
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            
            p_width = x2 - x1
            p_height = y2 - y1
            
            # Crash prevention
            if p_height == 0:
                continue
                
            aspect_ratio = p_width / p_height
            
            # False positive filter
            if aspect_ratio < 1.5 or aspect_ratio > 8.0:
                continue 

            plate_crop = frame[y1:y2, x1:x2]
            stable_text = recognize_plate(plate_crop)

            # Store text if valid
            if stable_text and stable_text not in extracted_plates_list:
                extracted_plates_list.append(stable_text)

            # --- 🎨 DRAWING LOGIC STARTS HERE ---
            # 1. Draw small RED rectangle tightly around license plate
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # 2. Estimate a larger GREEN bounding box around the vehicle
            car_x1 = max(0, x1 - int(p_width * 1.3))
            car_y1 = max(0, y1 - int(p_height * 4.5))
            car_x2 = min(frame.shape[1], x2 + int(p_width * 1.3))
            car_y2 = min(frame.shape[0], y2 + int(p_height * 1.5))
            cv2.rectangle(frame, (car_x1, car_y1), (car_x2, car_y2), (0, 255, 0), 4)

            # 3. Formatted Zoom Overlay Logic (White text box)
            if plate_crop.size > 0:
                overlay_h, overlay_w = 110, 280
                plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))

                oy1 = car_y1 - overlay_h - 70
                ox1 = car_x1 + 10
                
                for (ax1, ay1, ax2, ay2) in active_overlays:
                    if not (ox1 + overlay_w < ax1 or ox1 > ax2 or oy1 + (overlay_h + 60) < ay1 or oy1 > ay2):
                        ox1 = car_x2 + 20  
                        break
                
                oy1 = max(70, min(oy1, frame.shape[0] - overlay_h - 10))
                ox1 = max(10, min(ox1, frame.shape[1] - overlay_w - 10))
                oy2, ox2 = oy1 + overlay_h, ox1 + overlay_w

                active_overlays.append((ox1, oy1 - 60, ox2, oy2))

                if oy2 <= frame.shape[0] and ox2 <= frame.shape[1]:
                    frame[oy1:oy2, ox1:ox2] = plate_resized
                    text_bg_y1 = oy1 - 55
                    
                    # Draw the solid white block
                    cv2.rectangle(frame, (ox1, text_bg_y1), (ox2, oy1), (255, 255, 255), -1)

                    # Print the black text onto the white block
                    if stable_text:
                        cv2.putText(frame, stable_text, (ox1 + 30, oy1 - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 3, cv2.LINE_AA)

    # Save the visual layout image
    cv2.imwrite(output_image_path, frame)
    
    # Save the text out into a clean plain text file
    with open(output_text_path, "w") as f:
        for plate in extracted_plates_list:
            f.write(plate + "\n")
            
    print("\n================ EXTRACTION REPORT ================")
    print(f"📸 Annotated Image saved to: '{output_image_path}'")
    print(f"📝 Extracted Text saved to:  '{output_text_path}'")
    for i, plate in enumerate(extracted_plates_list, 1):
        print(f"   {i}. Vehicle License Plate -> {plate}")
    print("===================================================\n")

    # Show the final image on your screen
    cv2.imshow("Target Image Presentation", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()