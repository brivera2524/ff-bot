import argparse
import cv2
import numpy as np
from mss import mss
import keyboard
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque
from pathlib import Path

# Note hit region
SCREENSHOT_REGION = (1330, 900, 800, 65) #x,y,w,h

INPUT_DELAY = 0.12
MATCH_THRESHOLD = 0.55
DEFAULT_KEYS = 'dfjkl'
DEFAULT_WORKERS = 5
DEFAULT_PREVIEW_EVERY = 1
DEFAULT_PAUSE_KEY = 'p'
DEFAULT_QUIT_KEY = 'q'
TEMPLATE_DIR = Path(__file__).parent / "templates"
PRESS_TEMPLATE_PATH = TEMPLATE_DIR / "press_tight_crop.png"
LIFT_TEMPLATE_PATH = TEMPLATE_DIR / "lift_tight_crop.png"
pending_actions = deque()

def parse_region(value):
    parts = value.split(',')
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be x,y,width,height")

    try:
        x, y, w, h = [int(part.strip()) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("region values must be integers") from exc

    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("region width and height must be positive")

    return x, y, w, h

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fortnite Festival computer-vision bot."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="show the detection preview window",
    )
    parser.add_argument(
        "--no-topmost",
        action="store_true",
        help="do not keep the preview window always on top",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=INPUT_DELAY,
        help=f"input delay in seconds (default: {INPUT_DELAY})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help=f"template match confidence threshold (default: {MATCH_THRESHOLD})",
    )
    parser.add_argument(
        "--region",
        type=parse_region,
        default=SCREENSHOT_REGION,
        help="capture region as x,y,width,height",
    )
    parser.add_argument(
        "--keys",
        default=DEFAULT_KEYS,
        help=f"lane keys from left to right (default: {DEFAULT_KEYS})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"lane detection worker threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=DEFAULT_PREVIEW_EVERY,
        help=f"refresh the preview every N frames (default: {DEFAULT_PREVIEW_EVERY})",
    )
    parser.add_argument(
        "--pause-key",
        default=DEFAULT_PAUSE_KEY,
        help=f"toggle pause/resume key (default: {DEFAULT_PAUSE_KEY})",
    )
    parser.add_argument(
        "--quit-key",
        default=DEFAULT_QUIT_KEY,
        help=f"quit key (default: {DEFAULT_QUIT_KEY})",
    )
    parser.add_argument(
        "--start-active",
        action="store_true",
        help="start detecting immediately instead of launching paused",
    )

    args = parser.parse_args(argv)

    if len(args.keys) != 5:
        parser.error("--keys must contain exactly 5 characters")
    if len(set(args.keys)) != 5:
        parser.error("--keys must not contain duplicate characters")
    if args.delay < 0:
        parser.error("--delay must be 0 or greater")
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.workers <= 0:
        parser.error("--workers must be greater than 0")
    if args.preview_every <= 0:
        parser.error("--preview-every must be greater than 0")
    if len(args.pause_key) != 1:
        parser.error("--pause-key must be a single character")
    if len(args.quit_key) != 1:
        parser.error("--quit-key must be a single character")
    if args.pause_key == args.quit_key:
        parser.error("--pause-key and --quit-key must be different")
    if args.pause_key in args.keys or args.quit_key in args.keys:
        parser.error("--pause-key and --quit-key must not overlap with lane keys")

    return args

def load_template_image(path):
    template_bgr = cv2.imread(str(path))
    if template_bgr is None:
        raise FileNotFoundError(f"Missing template image: {path}")
    return cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

def load_lift_template(path):
    template_gray = load_template_image(path)
    
    h, w = template_gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    
    mid = h // 2

    triangle = np.array([[w//2, 0], [0, mid], [w, mid]])
    cv2.fillPoly(mask, [triangle], 255)
    cv2.rectangle(mask, (0, mid), (w, h), 255, -1)

    return template_gray, mask

def load_press_template(path):
    return load_template_image(path)

def build_lane_states(keys):
    return {key: 1 for key in keys}

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

def process_actions(input_delay):
    now = time.time()
    while pending_actions and now - pending_actions[0][0] >= input_delay:
        _, action, key = pending_actions.popleft()
        if action == 'press':
            Thread(target=press_with_release, args=(key,), daemon=True).start()
        elif action == 'release':
            keyboard.release(key)

def set_lane_states_released(keys, states):
    pending_actions.clear()
    for key in keys:
        keyboard.release(key)
        states[key] = 1

def toggle_pause_if_requested(paused, pause_key, keys, states, last_pause_press):
    pause_pressed = keyboard.is_pressed(pause_key)

    if pause_pressed and not last_pause_press:
        paused = not paused
        if paused:
            set_lane_states_released(keys, states)
        print("Paused" if paused else "Running")

    return paused, pause_pressed

def should_stop(preview_enabled, quit_key):
    if preview_enabled:
        cv2.waitKey(1)

    return keyboard.is_pressed(quit_key)

def draw_pause_overlay(screenshot, paused):
    if not paused:
        return

    cv2.rectangle(screenshot, (0, 0), (180, 34), (0, 0, 0), -1)
    cv2.putText(
        screenshot,
        "PAUSED",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )

def main():
    args = parse_args()
    keys = args.keys

    press_template = load_press_template(PRESS_TEMPLATE_PATH)
    lift_template, lift_mask = load_lift_template(LIFT_TEMPLATE_PATH)   
    press_w, press_h = press_template.shape[::-1]
    lift_w, lift_h = lift_template.shape[::-1]

    executor = ThreadPoolExecutor(max_workers=args.workers)
    x, y, w, h = args.region
    monitor = {"left": x, "top": y, "width": w, "height": h}

    # pre-compute region boundaries
    region_boundaries = [(i * w // 5, (i + 1) * w // 5) for i in range(5)]

    states = build_lane_states(keys)

    frame_count = 0
    paused = not args.start_active
    last_pause_press = False
    print("Paused" if paused else "Running")

    for _ in range(args.workers):
        executor.submit(lambda: None)
        time.sleep(0.1)  # let threads spin up

    if args.preview:
        cv2.namedWindow("preview", cv2.WINDOW_NORMAL)
        if not args.no_topmost:
            cv2.setWindowProperty("preview", cv2.WND_PROP_TOPMOST, 1)

    try:
        with mss() as sct:
            while True:
                screenshot = np.array(sct.grab(monitor))
                screenshot = np.ascontiguousarray(screenshot[:, :, :3])

                paused, last_pause_press = toggle_pause_if_requested(
                    paused,
                    args.pause_key,
                    keys,
                    states,
                    last_pause_press
                )
                
                if paused:
                    results = [None] * len(keys)
                else:
                    # use pre-computed boundaries instead of array_split
                    note_regions_gray = [
                        cv2.cvtColor(screenshot[:, x1:x2], cv2.COLOR_BGR2GRAY)
                        for x1, x2 in region_boundaries
                    ]

                    # detect all 5 regions in parallel
                    results = list(executor.map(
                        lambda i: detect_notes(
                            note_regions_gray[i],
                            press_template,
                            lift_template,
                            lift_mask,
                            args.threshold
                        ),
                        range(5)
                    ))

                    for i, result in enumerate(results):
                        current_key = keys[i]
                        handle_inputs(current_key, result, states)

                    process_actions(args.delay)

                frame_count += 1
                if args.preview and frame_count % args.preview_every == 0:
                    for i, result in enumerate(results):
                        if result is not None:
                            x1, _ = region_boundaries[i]
                            local_x, local_y = result['loc']
                            top_left = (local_x + x1, local_y)
                            
                            if result['type'] == "press":
                                cv2.rectangle(screenshot, top_left, (top_left[0] + press_w, top_left[1] + press_h), (0, 0, 255), 5)
                            else:
                                cv2.rectangle(screenshot, top_left, (top_left[0] + lift_w, top_left[1] + lift_h), (255, 0, 0), 5)

                    for _, x2 in region_boundaries:
                        cv2.line(screenshot, (x2, 0), (x2, screenshot.shape[0] - 1), (0, 0, 0), 1)

                    draw_pause_overlay(screenshot, paused)
                    cv2.imshow("preview", screenshot)
                    if not args.no_topmost:
                        cv2.setWindowProperty("preview", cv2.WND_PROP_TOPMOST, 1)

                if should_stop(args.preview, args.quit_key):
                    break
    finally:
        set_lane_states_released(keys, states)
        cv2.destroyAllWindows()
        executor.shutdown()

if __name__ == "__main__":
    main()
