import cv2
import numpy as np
from mss import mss
import keyboard
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque

# Note hit region
SCREENSHOT_REGION = (1330, 900, 800, 65) #x,y,w,h

INPUT_DELAY = 0.12
pending_actions = deque()
executor = ThreadPoolExecutor(max_workers=5)

def load_lift_template(path):
    template_bgr = cv2.imread(path)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    h, w = template_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    mid = h // 2

    triangle = np.array([[w//2, 0], [0, mid], [w, mid]])
    cv2.fillPoly(mask, [triangle], 255)
    cv2.rectangle(mask, (0, mid), (w, h), 255, -1)

    return template_gray, mask

def load_press_template(path):
    template_bgr = cv2.imread(path)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    return template_gray

def detect_notes(img, note_template, lift_template, mask, threshold=0.55, method=cv2.TM_CCOEFF_NORMED):
    lift_mask = None

    for template, note_type in [(note_template, 'press'), (lift_template, 'lift')]:
        if note_type == 'lift':
            lift_mask = mask
        res = cv2.matchTemplate(image=img, templ=template, method=method, mask=lift_mask)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val >= threshold:
            return {'type': note_type, 'loc': max_loc, 'confidence': max_val}

    return None

def press_with_release(key):
    keyboard.release(key)
    time.sleep(0.01)
    keyboard.press(key)

def handle_inputs(key, result, states):
    current_key_state = states[key]
    
    if result is not None:
        note_type = result['type']
        
        if note_type == 'press' and current_key_state == 1:
            pending_actions.append((time.time(), 'press', key))
            states[key] = 0
            
        elif note_type == 'lift':
            pending_actions.append((time.time(), 'release', key))
            states[key] = 1
    else:
        states[key] = 1

def process_actions():
    now = time.time()
    while pending_actions and now - pending_actions[0][0] >= INPUT_DELAY:
        _, action, key = pending_actions.popleft()
        if action == 'press':
            Thread(target=press_with_release, args=(key,), daemon=True).start()
        elif action == 'release':
            keyboard.release(key)

def main():
    keys = 'dfjkl'
    
    press_template = load_press_template('templates/press_tight_crop.png')
    lift_template, lift_mask = load_lift_template('templates/lift_tight_crop.png')   
    
    press_w, press_h = press_template.shape[::-1]
    lift_w, lift_h = lift_template.shape[::-1]
    x, y, w, h = SCREENSHOT_REGION
    monitor = {"left": x, "top": y, "width": w, "height": h}

    # pre-compute region boundaries
    region_boundaries = [(i * w // 5, (i + 1) * w // 5) for i in range(5)]

    states = {
        'd': 1,
        'f': 1,
        'j': 1,
        'k': 1,
        'l': 1
    }

    frame_count = 0

    for _ in range(5):
        executor.submit(lambda: None)
        time.sleep(0.1)  # let threads spin up

    with mss() as sct:
        while True:
            screenshot = np.array(sct.grab(monitor))
            screenshot = np.ascontiguousarray(screenshot[:, :, :3])
            
            # use pre-computed boundaries instead of array_split
            note_regions_gray = [
                cv2.cvtColor(screenshot[:, x1:x2], cv2.COLOR_BGR2GRAY)
                for x1, x2 in region_boundaries
            ]

            # detect all 5 regions in parallel
            results = list(executor.map(
                lambda i: detect_notes(note_regions_gray[i], press_template, lift_template, lift_mask),
                range(5)
            ))

            for i, result in enumerate(results):
                current_key = keys[i]
                handle_inputs(current_key, result, states)

            process_actions()

            # only draw preview every 3 frames
            frame_count += 1
            if frame_count % 3 == 0:
                for i, result in enumerate(results):
                    if result is not None:
                        x1, _ = region_boundaries[i]
                        local_x, local_y = result['loc']
                        top_left = (local_x + x1, local_y)
                        
                        if result['type'] == "press":
                            cv2.rectangle(screenshot, top_left, (top_left[0] + press_w, top_left[1] + press_h), (0, 0, 255), 5)
                        else:
                            cv2.rectangle(screenshot, top_left, (top_left[0] + lift_w, top_left[1] + lift_h), (255, 0, 0), 5)

                x_offset = 0
                for x1, x2 in region_boundaries:
                    cv2.line(screenshot, (x2, 0), (x2, screenshot.shape[0] - 1), (0, 0, 0), 1)

    
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    executor.shutdown()

if __name__ == "__main__":
    main()