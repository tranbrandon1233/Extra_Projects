import cv2
import numpy as np
import pyaudio
import struct
import time

# Video Settings
VIDEO_FILE = 'video.mp4'
OUTPUT_VIDEO_FILE = ''
# Audio Settings
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
FRAMES_PER_BUFFER = 1024

# Color Settings
COLORS = [(255, 0, 0), (255, 128, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (128, 0, 255)]

class AudioVisualizer:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  frames_per_buffer=FRAMES_PER_BUFFER)
        self.data = np.zeros(FRAMES_PER_BUFFER)
        self.color_index = 0

    def get_audio_data(self):
        data = np.frombuffer(self.stream.read(FRAMES_PER_BUFFER), dtype=np.int16)
        self.data = np.abs(np.fft.rfft(data).real)

    def draw_visualizer(self, frame):
        self.get_audio_data()
        data = self.data / max(self.data) * frame.shape[0]
        for i in range(len(data)):
            color = COLORS[(i + self.color_index) % len(COLORS)]
            cv2.line(frame, (int(i * frame.shape[1] / len(data)), frame.shape[0]), (int(i * frame.shape[1] / len(data)), frame.shape[0] - int(data[i])), color, 2)
        self.color_index = (self.color_index + 1) % len(COLORS)

def main():
    visualizer = AudioVisualizer()
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        visualizer.draw_visualizer(frame)
        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()