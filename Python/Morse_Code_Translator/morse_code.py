import os
from pydub import AudioSegment


MORSE_CODE_DICT = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 'F': '..-.', 'G': '--.',
    'H': '....', 'I': '..', 'J': '.---', 'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.',
    'O': '---', 'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 'U': '..-',
    'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 'Z': '--..', '0': '-----',
    '1': '.----', '2': '..---', '3': '...--', '4': '....-', '5': '.....', '6': '-....',
    '7': '--...', '8': '---..', '9': '----.', '.': '.-.-.-', ',': '--..--', '?': '..--..',
    '/': '-..-', '-': '-....-', '(': '-.--.', ')': '-.--.-', ' ': '/'
}

def convert_sentence_to_morse(sentence):
    """
    Converts a sentence to Morse code, separating words with '/'.

    Args:
        sentence (str): The sentence to convert.

    Returns:
        str: The Morse code representation of the sentence.
    """

    morse_words = []
    for word in sentence.upper().split():
        morse_word = ''.join(MORSE_CODE_DICT.get(char, '') for char in word)
        morse_words.append(morse_word)
    return '/'.join(morse_words)
    

DOT_DURATION = 200  # Duration of a dot in milliseconds
DASH_DURATION = DOT_DURATION * 3
SPACE_DURATION = DOT_DURATION

# Generate audio segments for dot, dash, and space
dot_sound = AudioSegment.from_file('beep.mp3')[:DOT_DURATION]
dash_sound = AudioSegment.from_file('beep.mp3')[:DASH_DURATION]
space_sound = AudioSegment.silent(SPACE_DURATION)

def generate_morse_code_sound(morse_code):
    """
    Generates Morse code sound for a given Morse code string.

    Args:
        morse_code (str): The Morse code string.

    Returns:
        pydub.AudioSegment: The audio segment representing the Morse code sound.
    """

    audio = AudioSegment.empty()
    for symbol in morse_code:
        if symbol == '.':
            audio += dot_sound
        elif symbol == '-':
            audio += dash_sound
        elif symbol == '/':
            audio += space_sound * 3  # Word separation is 7 units, so 3 spaces + 1 space between symbols
        else:
            audio += space_sound  # Space between symbols within a letter (1 unit)
    return audio

# Example usage
# Replace the audio file paths and sentence as needed.
sentence = "Hello World"
morse_code = convert_sentence_to_morse(sentence)
morse_sound = generate_morse_code_sound(morse_code)
morse_sound.export("morse_code_sound.wav", format="wav")
