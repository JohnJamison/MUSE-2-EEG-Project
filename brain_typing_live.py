import time
import numpy as np
import keyboard
import joblib

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations

MODEL_PATH = "brain_control_model.pkl"
LABEL_ENCODER_PATH = "brain_label_encoder.pkl"
SERIAL = "Muse-2919"

WINDOW_SEC = 3.0
STEP_SEC = 1.0
NOTCH_FREQ = 60.0
BANDPASS_LOW = 1.0
BANDPASS_HIGH = 45.0

print("\n🧠 Loading model...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)
print("✅ Model loaded:", label_encoder.classes_, "\n")

params = BrainFlowInputParams()
params.serial_number = SERIAL
board = BoardShim(BoardIds.MUSE_2_BOARD.value, params)

print("🔌 Connecting to Muse...")
board.prepare_session()
board.start_stream()
time.sleep(3)
print("✅ Streaming started\n")

fs = BoardShim.get_sampling_rate(BoardIds.MUSE_2_BOARD.value)
eeg_ch = BoardShim.get_eeg_channels(BoardIds.MUSE_2_BOARD.value)

samples_window = int(WINDOW_SEC * fs)
buffer = np.empty((len(eeg_ch), 0))  # Keep growing & sliding


def compute_band(sig, low, high):
    psd, freqs = DataFilter.get_psd_welch(
        sig.astype(np.float64),
        nfft=256,
        overlap=128,
        sampling_rate=fs,
        window=WindowOperations.HANNING.value
    )
    idx = np.logical_and(freqs >= low, freqs <= high)
    return float(np.trapezoid(psd[idx], freqs[idx]))


def extract_features(window):
    alpha_list, beta_list, theta_list, gamma_list = [], [], [], []

    for ch_index in range(window.shape[0]):
        sig = window[ch_index, :].copy()

        DataFilter.detrend(sig, DetrendOperations.LINEAR.value)
        DataFilter.perform_bandpass(sig, fs, BANDPASS_LOW, BANDPASS_HIGH, 4, FilterTypes.BUTTERWORTH.value, 0)
        DataFilter.perform_bandstop(sig, fs, NOTCH_FREQ - 1, NOTCH_FREQ + 1, 3, FilterTypes.BUTTERWORTH.value, 0)

        alpha_list.append(compute_band(sig, 8, 12))
        beta_list.append(compute_band(sig, 13, 30))
        theta_list.append(compute_band(sig, 4, 7))
        gamma_list.append(compute_band(sig, 31, 45))

    alpha, beta, theta, gamma = map(np.mean, (alpha_list, beta_list, theta_list, gamma_list))

    eps = 1e-9
    return np.array([[alpha, beta, theta, gamma,
                      (beta+eps)/(alpha+eps),
                      (alpha+eps)/(theta+eps),
                      (gamma+eps)/(beta+eps),
                      np.log10(alpha+eps),
                      np.log10(beta+eps),
                      np.log10(theta+eps),
                      np.log10(gamma+eps)
                      ]], dtype=np.float64)


def send_action(pred):
    if pred == "right":
        keyboard.write("Right ")
    elif pred == "left":
        keyboard.write("Left ")
    elif pred == "down":
        keyboard.write("Down ")
    elif pred == "up":
        keyboard.write("Up ")


print("🎯 Think LEFT / RIGHT — neutral = relax completely")
print("CTRL+C to exit.\n")


try:
    while True:
        chunk = board.get_current_board_data(40)
        if chunk.size > 0:
            buffer = np.hstack((buffer, chunk[eeg_ch, :]))
            if buffer.shape[1] > fs*4:  # Keep +4 seconds max
                buffer = buffer[:, -fs*4:]

        if buffer.shape[1] >= samples_window:
            window = buffer[:, -samples_window:]
            feats = extract_features(window)
            probs = model.predict_proba(feats)[0]
            pred = label_encoder.inverse_transform([np.argmax(probs)])[0]
            conf = float(np.max(probs))

            # ✅ ALWAYS PRINT PREDICTION
            print(f"> {pred.upper():7}  conf={conf:.2f}  raw={np.round(probs,3)}")

            # ✅ ALWAYS PERFORM ACTION
            send_action(pred)

        else:
            print(".", end="", flush=True)

        time.sleep(0.05)

except KeyboardInterrupt:
    print("\n🛑 Stopping stream...")

finally:
    board.stop_stream()
    board.release_session()
    print("✅ Muse disconnected.")
