import cv2
import numpy as np
from mss import mss
from pynput.keyboard import Controller

# Note hit region
SCREENSHOT_REGION = (1245, 1054, 950, 150)


def load_lift_template(path):
    template_bgr = cv2.imread(path)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    h, w = template_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)  # full height
    
    mid = h // 2

    triangle = np.array([[w//2, 0], [0, mid], [w, mid]])
    cv2.fillPoly(mask, [triangle], 255)

    cv2.rectangle(mask, (0, mid), (w, h), 255, -1)

    return template_gray, mask

def load_press_template(path):
    template_bgr = cv2.imread(path)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    return template_gray

def detect_notes(img, note_template, lift_template, mask, threshold=0.8, method=cv2.TM_CCOEFF_NORMED):
    img = img.copy()
    lift_mask = None

    for template, note_type in [(note_template, 'press'), (lift_template, 'lift')]:
        if note_type == 'lift':
            lift_mask = mask
        res = cv2.matchTemplate(image=img, templ=template, method=method, mask=lift_mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            return {'type': note_type, 'loc': max_loc, 'confidence': max_val}

    return None



def main():
    
    keys = 'dfjkl'
    
    press_template = load_press_template('templates/press.png')
    lift_template, lift_mask = load_lift_template('templates/lift_right.png')   
    
    press_w, press_h= press_template.shape[::-1]
    lift_w, lift_h= lift_template.shape[::-1]
    x, y, w, h = SCREENSHOT_REGION
    monitor = {"left": x, "top": y, "width": w, "height": h}

    keyboard = Controller()
    

    with mss() as sct:
        while True:
            screenshot = np.array(sct.grab(monitor))
            screenshot = np.ascontiguousarray(screenshot[:, :, :3])
            
            # Split image into 5 parts
            note_regions = np.array_split(screenshot, 5, axis=1)
            note_regions_gray = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in note_regions]

            # TODO: no need to calculate offset every time. Should calculate boundaries once and store.
            x_offset = 0
            
            # Process one region at a time. Multithread possible?
            for i, region in enumerate(note_regions_gray):
                
                current_key = keys[i]
                
                result = detect_notes(region, press_template, lift_template, lift_mask)
                
                if result is not None:
                    # offset coords to align with unsplit image    
                    local_x, local_y = result['loc']
                    top_left = (local_x + x_offset, local_y)
                    
                    if result['type'] == "press":
                    
                                            
                        # draw red box around press note
                        cv2.rectangle(
                            screenshot,
                            top_left,
                            (top_left[0] + press_w, top_left[1] + press_h),
                            (0, 0, 255),
                            5
                        )
                    else:
                        cv2.rectangle(
                            screenshot,
                            top_left,
                            (top_left[0] + lift_w, top_left[1] + lift_h),
                            (255, 0, 0),
                            5
                        )
                    
                    
                    

                    print(result['confidence'], result['type'])
                    
                x_offset += region.shape[1]
                    
                cv2.line(
                    screenshot,
                    (x_offset, 0),
                    (x_offset, screenshot.shape[0] - 1),
                    (0, 0, 0),
                    1
                )
                    
            cv2.imshow("preview", screenshot)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()