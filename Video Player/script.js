// script.js
const fileInput = document.getElementById('file-input');
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

fileInput.addEventListener('change', async (e) => {
    const file = fileInput.files[0];
    const url = URL.createObjectURL(file);
    video.src = url;

    // Wait for the video to load
    await new Promise((resolve) => {
        video.addEventListener('loadeddata', resolve);
    });

    // Extract the audio from the video file
    const audioContext = new AudioContext();
    const source = audioContext.createMediaElementSource(video);
    const gain = audioContext.createGain();
    source.connect(gain);
    gain.connect(audioContext.destination);

    // Use the Google Cloud Speech-to-Text API to recognize speech from the audio
    const apiKey = '';
    const apiEndpoint = 'https://speech.googleapis.com/v1/speech:recognize?key=' + apiKey;
    const audioData = await getAudioData(video, audioContext);
    const response = await fetch(apiEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            config: {
                encoding: 'LINEAR16',
                sampleRateHertz: 48000,
                languageCode: 'en-US',
                enableWordTimeOffsets: true,
            },
            audio: {
                content: audioData,
            },
        }),
    });
    const data = await response.json();
    const transcript = data.results[0].alternatives[0].transcript;
    const timestamps = data.results[0].alternatives[0].words.map((word) => {
        return { time: word.startTime.seconds + word.startTime.nanos / 1e9, text: word.word };
    });

    // Display the captions
    displayCaptions(transcript, timestamps, video);
});

// Get the audio data from the video file
function getAudioData(video, audioContext) {
    return new Promise((resolve) => {
        const audioData = [];
        const scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
            audioData.push(e.inputBuffer.getChannelData(0));
        };
        const source = audioContext.createMediaElementSource(video);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        video.addEventListener('ended', () => {
            const audioDataArray = new Float32Array(audioData.length * 1024);
            for (let i = 0; i < audioData.length; i++) {
                audioDataArray.set(audioData[i], i * 1024);
            }
            const audioDataInt16 = new Int16Array(audioDataArray.length);
            for (let i = 0; i < audioDataArray.length; i++) {
                audioDataInt16[i] = audioDataArray[i] * 32767;
            }
            const audioDataBase64 = btoa(String.fromCharCode(...new Uint8Array(audioDataInt16.buffer)));
            resolve(audioDataBase64);
        });
    });
}

// Display the captions
function displayCaptions(transcript, timestamps, video) {
    let index = 0;
    let previousTime = 0;
    video.addEventListener('timeupdate', () => {
        const time = video.currentTime;
        if (index < timestamps.length && time >= timestamps[index].time && time > previousTime) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillText(timestamps[index].text, 10, 30);
            index++;
        }
        previousTime = time;
    });
    video.addEventListener('pause', () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
    });
    video.addEventListener('seeked', () => {
        index = 0;
        previousTime = 0;
    });
}