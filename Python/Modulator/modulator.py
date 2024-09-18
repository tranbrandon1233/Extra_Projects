import numpy as np
import sounddevice as sd
import time

# --- Configuration ---
CARRIER_FREQ = 1000  # Hz
SYMBOL_RATE = 100  # Baud
FFT_SIZE = 1024
CP_LENGTH = FFT_SIZE // 4
PSK_ORDER = 2  # BPSK

# --- Functions ---
def modulate_psk(data, order):
    """Modulate data using PSK."""
    symbols = np.array(data).reshape(-1, 1)
    phase_shifts = np.exp(1j * 2 * np.pi * np.arange(order) / order)
    modulated = phase_shifts[symbols % order]
    return modulated.flatten()

def demodulate_psk(symbols, order):
    """Demodulate PSK symbols."""
    phase_shifts = np.exp(1j * 2 * np.pi * np.arange(order) / order)
    distances = np.abs(symbols[:, None] - phase_shifts[None, :])
    detected_symbols = np.argmin(distances, axis=1)
    return detected_symbols

def ofdm_modulate(data, fft_size, cp_length):
    """Modulate data using OFDM."""
    # Pad data to be a multiple of FFT size
    padding = (fft_size - len(data) % fft_size) % fft_size
    data = np.concatenate((data, np.zeros(padding)))

    # Reshape data into OFDM symbols
    symbols = data.reshape(-1, fft_size)

    # Perform IFFT
    symbols_time = np.fft.ifft(symbols, axis=1)

    # Add cyclic prefix
    symbols_cp = np.concatenate((symbols_time[:, -cp_length:], symbols_time), axis=1)

    # Flatten and return
    return symbols_cp.flatten()

def ofdm_demodulate(signal, fft_size, cp_length):
    """Demodulate OFDM signal."""
    # Reshape signal into OFDM symbols
    symbols_cp = signal.reshape(-1, fft_size + cp_length)

    # Remove cyclic prefix
    symbols_time = symbols_cp[:, cp_length:]

    # Perform FFT
    symbols = np.fft.fft(symbols_time, axis=1)

    # Extract data and return
    return symbols.flatten()

def encode_and_transmit(data):
    """Encode data using PSK-OFDM and transmit over audio."""
    # Convert data to binary representation
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))

    # Modulate using PSK
    modulated_symbols = modulate_psk(bits, PSK_ORDER)

    # Modulate using OFDM
    ofdm_signal = ofdm_modulate(modulated_symbols, FFT_SIZE, CP_LENGTH)

    # Normalize to [-1, 1]
    ofdm_signal /= np.max(np.abs(ofdm_signal))

    # Generate carrier wave
    t = np.arange(len(ofdm_signal)) / SYMBOL_RATE
    carrier = np.sin(2 * np.pi * CARRIER_FREQ * t)

    # Modulate with carrier
    tx_signal = ofdm_signal * carrier

    # Play audio
    sd.play(tx_signal, samplerate=44100)
    sd.wait()

def receive_and_decode():
    """Receive audio, demodulate PSK-OFDM, and return decoded data."""
    # Record audio
    print("Recording...")
    recording = sd.rec(int(5 * 44100), samplerate=44100, channels=1)
    sd.wait()
    print("Recording finished.")

    # Demodulate with carrier (assuming perfect synchronization for simplicity)
    t = np.arange(len(recording)) / 44100
    carrier = np.sin(2 * np.pi * CARRIER_FREQ * t)
    rx_signal = recording[:, 0] * carrier

    # Demodulate using OFDM
    demodulated_symbols = ofdm_demodulate(rx_signal, FFT_SIZE, CP_LENGTH)

    # Demodulate using PSK
    received_bits = demodulate_psk(demodulated_symbols, PSK_ORDER)

    # Convert bits to bytes
    received_data = np.packbits(received_bits).tobytes()

    return received_data

# --- Main ---
if __name__ == "__main__":
    # Get data from stdin
    input_data = input("Enter data to transmit: ").encode()

    # Encode and transmit
    encode_and_transmit(input_data)

    # Receive and decode
    received_data = receive_and_decode()

    # Print received data to stdout
    print("Received data:", received_data.decode())