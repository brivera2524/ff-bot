# Real-Time Computer Vision Input Automation

Real-time computer-vision automation that detects on-screen cues with OpenCV template matching and fires precision-timed keyboard inputs. The working application is a bot for Fortnite Festival, a five-lane rhythm game: it watches the note highway, detects incoming press and lift notes, and plays them with tuned timing.

This started as a practical automation experiment: fast screen capture, per-lane image processing, input scheduling, and performance-minded Python threading all working together in a real-time environment with hard latency constraints.

![Detection demo](assets/demo.gif)

## What It Does

- Captures a fixed region of the screen where notes appear.
- Splits the note highway into five lanes mapped to `D`, `F`, `J`, `K`, and `L`.
- Uses OpenCV template matching to detect standard press notes and lift notes.
- Applies a custom mask for lift-note detection to reduce false positives.
- Processes all five lanes in parallel with a thread pool.
- Queues key press and release actions with a configurable timing delay.
- Maintains lane state so held notes and releases are handled cleanly.

## Tech Stack

- Python
- OpenCV
- NumPy
- MSS screen capture
- Keyboard input automation
- ThreadPoolExecutor for parallel lane detection

## How It Works

The bot continuously grabs a small screen region around the note highway. Each frame is converted into five grayscale lane regions. For each lane, the bot compares the live image against saved note templates in `templates/`:

- `press_tight_crop.png` for standard notes
- `lift_tight_crop.png` for lift notes

When a note is detected above the confidence threshold, the bot schedules the matching keyboard action. Press notes briefly reset and press the lane key, while lift notes release it. The action queue adds a short delay so detection and input timing can be tuned for latency. Keys are held down until another key or a lift note is detected in the same lane, allowing the bot to score maximum points on sustained notes.

## Project Structure

```text
.
|-- assets/
|   `-- demo.gif              # Project demo
|-- main.py                  # Main screen capture, detection, and input loop
|-- templates/               # Note templates used by OpenCV matching
|   |-- press_tight_crop.png
|   `-- lift_tight_crop.png
|-- tests/                   # Unit tests for CLI parsing and template helpers
`-- requirements.txt
```

## Setup

This project targets Windows because it uses keyboard input automation against a running application window.

1. Clone the repo:

```powershell
   git clone https://github.com/brivera2524/realtime-cv-automation.git
   cd realtime-cv-automation
```

2. Create and activate a virtual environment:

```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
   pip install -r requirements.txt
```

4. Start Fortnite Festival and position the game so the note highway matches the configured capture region in `main.py`.

5. Run the bot:

```powershell
   python main.py
```

The bot launches paused by default. Switch to the game, then press `p` to start detecting and inputting notes. Press `p` again to pause, or `q` to quit.

To run with the detection preview window:

```powershell
python main.py --preview
```

The preview window stays always on top by default. To show the preview without forcing it above other windows:

```powershell
python main.py --preview --no-topmost
```

To start detecting immediately instead of launching paused:

```powershell
python main.py --start-active
```

## Configuration

The main calibration point is the screenshot region near the top of `main.py`:

```python
SCREENSHOT_REGION = (1330, 900, 800, 65) # x, y, width, height
```

If the bot is not detecting notes, update this tuple to match the note highway on your monitor. The values are screen coordinates for the region the bot captures and analyzes.

You can also tune:

- `INPUT_DELAY` for timing calibration
- the template matching threshold in `detect_notes`
- the lane key mapping with `--keys`

These values can also be changed from the command line:

```powershell
python main.py --delay 0.10 --threshold 0.60 --region 1330,900,800,65 --keys dfjkl
```

Available CLI options:

- `--preview` shows the OpenCV detection window.
- `--no-topmost` disables always-on-top behavior for the preview window.
- `--delay` sets the input delay in seconds.
- `--threshold` sets the OpenCV template match confidence threshold.
- `--region` sets the capture box as `x,y,width,height`.
- `--keys` sets the five lane keys from left to right.
- `--workers` sets the number of detection worker threads.
- `--preview-every` controls how often the preview window refreshes. The default is every frame.
- `--pause-key` changes the pause/resume key. The default is `p`.
- `--quit-key` changes the quit key. The default is `q`.
- `--start-active` starts detection immediately instead of launching paused.

## Testing

The tests focus on the pure Python pieces that can run without the game open: CLI validation, capture-region parsing, custom lane-key state, and template loading errors.

```powershell
python -m pytest
```

## Why This Project Matters

This project shows how I approach real-time automation problems:

- breaking a visual problem into small, testable regions
- using image templates and masks for reliable detection
- optimizing the loop by precomputing lane boundaries
- parallelizing CPU work without overcomplicating the design
- separating detection, state tracking, and input execution

It is a small project, but it touches the same concerns that show up in production software: latency, reliability, state management, debugging visibility, and iterative calibration.

## Notes

This was built as a personal computer-vision and automation project. Use it responsibly, and be aware that automating gameplay may violate a game's terms of service.