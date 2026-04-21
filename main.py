import cv2
import numpy as np
from mss import mss

# Note hit region
SCREENSHOT_REGION = (1245, 1054, 950, 150)


def detect_notes(img, note_template, lift_template, threshold=0.5, method=cv2.TM_CCOEFF_NORMED):
    img = img.copy()
    
    res = cv2.matchTemplate(image=img, templ=note_template, method=method)
    note_type = 'press'
    
    # TODO: check for lift notes if no press note is found. return type: 'lift'?
    
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    top_left = max_loc  # higher is better
    
    if max_val < threshold:
        return None
    else:
        return {'type': 'press', 'loc': top_left, 'confidence': max_val}


def main():
    note_template_raw = cv2.imread('templates/note.png')
    note_template_gray = cv2.cvtColor(note_template_raw, cv2.COLOR_RGB2GRAY)
    
    note_w, note_h = note_template_gray.shape[::-1]

    x, y, w, h = SCREENSHOT_REGION
    monitor = {"left": x, "top": y, "width": w, "height": h}


    with mss() as sct:
        while True:
            screenshot = np.array(sct.grab(monitor))
            screenshot = np.ascontiguousarray(screenshot[:, :, :3])
            
            # Split image into 5 parts
            note_regions = np.array_split(screenshot, 5, axis=1)
            note_regions_gray = [cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) for region in note_regions]

            x_offset = 0
            
            # Process one region at a time
            for i, region in enumerate(note_regions_gray):
                result = detect_notes(region, note_template_gray, lift_template=None, threshold=0.6)
                
                if result is not None:
                    if result['type'] == "press":
                        # implement press handling
                        pass
                    else:
                        pass
                    
                    # offset coords to align with unsplit image    
                    local_x, local_y = result['loc']
                    top_left = (local_x + x_offset, local_y)
                    
                    # draw red box around note
                    cv2.rectangle(
                        screenshot,
                        top_left,
                        (top_left[0] + note_w, top_left[1] + note_h),
                        (0, 0, 255),
                        5
                    )
                    
                    
                    print(result['confidence'])
                x_offset += region.shape[1]

            # draw segment boundaries
            x_offset = 0
            for region in note_regions[:-1]:
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