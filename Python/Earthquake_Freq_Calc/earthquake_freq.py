import numpy as np
from scipy.signal import welch

def calculate_corner_frequency(frequencies, power_spectrum, plot=False):
  """
  Calculates the corner frequency of an earthquake from its power spectrum.

  Args:
    frequencies: An array of frequencies.
    power_spectrum: An array of power spectral density values corresponding to the frequencies.
    plot: A boolean indicating whether to plot the power spectrum and corner frequency.

  Returns:
    The corner frequency in Hz.
  """

  # Smooth the power spectrum to reduce noise
  smoothed_spectrum = np.convolve(power_spectrum, np.ones(5)/5, mode='same')

  # Find the peak of the spectrum
  peak_index = np.argmax(smoothed_spectrum)
  peak_frequency = frequencies[peak_index]

  # Find the frequency at which the power drops to half the peak value
  half_power = smoothed_spectrum[peak_index] / 2
  corner_index = np.argmin(np.abs(smoothed_spectrum[:peak_index] - half_power))
  corner_frequency = frequencies[corner_index]

  if plot:
    import matplotlib.pyplot as plt
    plt.loglog(frequencies, smoothed_spectrum)
    plt.axvline(corner_frequency, color='r', linestyle='--', label='Corner Frequency')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density')
    plt.title('Power Spectrum and Corner Frequency')
    plt.legend()
    plt.show()

  return corner_frequency

# Example usage
data = np.loadtxt("earthquake_data.txt")  # Load data from file
sampling_rate = 100

# Calculate the power spectrum
frequencies, power_spectrum = welch(data, fs=sampling_rate)

# Calculate the corner frequency
corner_freq = calculate_corner_frequency(frequencies, power_spectrum, plot=True)

print(f"Corner Frequency: {corner_freq:.2f} Hz")