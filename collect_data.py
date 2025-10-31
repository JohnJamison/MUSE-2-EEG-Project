import os
import time
import csv
import numpy as np
import datetime
from pathlib import Path
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

# =========================================
# CONFIG
# =========================================
DEFAULT_CSV = "collected_eeg_data.csv"
SERIAL = "Muse-2919"  # Change if needed

FIXATION_DURATION = 1.0
CUE_DURATION = 1.0
IMAGERY_DURATION = 3.0
REST_DURATION = 1.5

LABELS = ["left", "right", "neutral"]


# =========================================
# Determine save file with user prompt
# =========================================
csv_name = input(f"Enter dataset filename (ENTER = {DEFAULT_CSV}): ").strip()
if csv_name == "":
    csv_name = DEFAULT_CSV

csv_path = Path(csv_name)
is_new_file = not csv_path.exists()

print(f"📁 Using dataset file: {csv_path}")


# =========================================
# Session Tracking
# =========================================
session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"🆕 Session ID: {session_id}")


# =========================================
# Header + Trial Continuation Logic
# =========================================
header = ["timestamp", "session_id", "trial", "label", "ch1", "ch2", "ch3", "ch4"]

if is_new_file:
    print("✅ Creating new file with header")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
    next_trial = 1
else:
    print("📌 Appending to existing file...")
    try:
        existing = np.genfromtxt(csv_path, delimiter=",", dtype=str, skip_header=1)
        if existing.size > 0:
            last_trial = existing[:, 2].astype(int).max()
            next_trial = last_trial + 1
        else:
            next_trial = 1
    except Exception:
        next_trial = 1

print(f"👉 Next trial will start at: {next_trial}")


# =========================================
# Connect Muse
# =========================================
params = BrainFlowInputParams()
params.serial_number = SERIAL
board = BoardShim(BoardIds.MUSE_2_BOARD.value, params)

print("\n🔌 Connecting to Muse...")
board.prepare_session()
board.start_stream()
time.sleep(3)
print("✅ Muse streaming!\n")


# =========================================
# Trial recording function
# =========================================
def record_trial(trial, label):
    print("\n-------------------")
    print(f"Trial {trial} | {label.upper()}")
    print("-------------------")

    # Fixation (neutral reset)
    print(" + Fixation")
    time.sleep(FIXATION_DURATION)

    # Cue (visual recognition only)
    print(f" → Cue: {label}")
    time.sleep(CUE_DURATION)

    # Motor imagery
    print(" 🧠 THINK NOW!")
    start_time = time.time()
    board.insert_marker(1)

    eeg_points = []
    while time.time() - start_time < IMAGERY_DURATION:
        data = board.get_current_board_data(1)
        if data.size > 0:
            eeg_points.append(data)
        time.sleep(0.01)

    print(" …REST…")
    time.sleep(REST_DURATION)

    if len(eeg_points) == 0:
        print("⚠️ No EEG collected — trial skipped")
        return

    eeg = np.hstack(eeg_points)
    eeg_channels = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value)
    eeg = eeg[eeg_channels, :]  # ALL EEG channels

    # Reduce to feature vector (mean per channel)
    sample = np.mean(eeg, axis=1)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.time(),
            session_id,
            trial,
            label,
            sample[0], sample[1], sample[2], sample[3]
        ])

    print(f"✅ Saved trial {trial}: {label}")


# =========================================
# Main Loop
# =========================================
trial_num = next_trial
print("✨ Ready — enter L/R/N for each trial, Q to stop.")

try:
    while True:
        cmd = input("\n[L]eft [R]ight [N]eutral [Q]uit >> ").strip().lower()

        if cmd == "l":
            record_trial(trial_num, "left")
        elif cmd == "r":
            record_trial(trial_num, "right")
        elif cmd == "n":
            record_trial(trial_num, "neutral")
        elif cmd == "q":
            break
        else:
            print("Invalid key. Try again.")
            continue

        trial_num += 1

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")

finally:
    print("\n🔻 Closing connection")
    board.stop_stream()
    board.release_session()
    print("✅ Disconnected")
    print(f"🎉 Session complete — {trial_num-1} total trials recorded")
