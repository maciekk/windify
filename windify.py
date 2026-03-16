import numpy as np
import scipy.signal as sig
import scipy.io.wavfile as wav

def simulate_wind_and_distance(input_file, output_file, ir_file=None):
    # 1. Load the Audio
    sr, audio = wav.read(input_file)

    # Normalize input audio to -1.0 to 1.0 float64
    if audio.dtype == np.int16:
        audio = audio.astype(np.float64) / 32768.0

    # Convert stereo to mono for easier processing if necessary
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    total_samples = len(audio)
    time = np.arange(total_samples) / sr

    # 2. Generate the "Wind" LFO
    # We combine a few low-frequency sine waves to create an unpredictable, rolling "gust" 
    # pattern rather than a perfect, robotic oscillation.
    lfo = (np.sin(2 * np.pi * 0.4 * time) + 
           np.sin(2 * np.pi * 0.15 * time + 0.5) + 
           np.sin(2 * np.pi * 0.05 * time + 1.2))

    # Normalize LFO to range [0.0, 1.0]
    wind_lfo = (lfo - np.min(lfo)) / (np.max(lfo) - np.min(lfo))

    # 3. Dynamic Filtering and Amplitude Modulation (Block Processing)
    frame_size = 1024
    output_audio = np.zeros_like(audio)

    # We need to maintain the filter state between frames so the audio doesn't click or pop
    # when the coefficients change slightly.
    zi = np.zeros(2) # State vector for a 2nd-order filter

    for i in range(0, total_samples, frame_size):
        # Extract the current frame
        frame = audio[i:i+frame_size]
        if len(frame) == 0:
            break

        # Get the average wind intensity for this specific frame
        current_wind = np.mean(wind_lfo[i:i+frame_size])

        # --- Amplitude Modulation ---
        # Base volume is low (distance). Wind gusts bring the volume up slightly.
        # Gain maps from 0.1 (lull) to 0.4 (gust)
        gain = 0.1 + (0.3 * current_wind)
        frame_amped = frame * gain

        # --- Dynamic Low-Pass Filter ---
        # Distance inherently cuts highs. Wind pushes higher frequencies toward us.
        # Cutoff frequency $f_c$ maps from 400Hz (lull) to 1200Hz (gust)
        cutoff_hz = 400.0 + (800.0 * current_wind)

        # Normalize cutoff frequency to the Nyquist frequency (sr / 2)
        w_n = cutoff_hz / (sr / 2.0)

        # Generate 2nd-order Butterworth filter coefficients for this frame
        b, a = sig.butter(2, w_n, btype='low')

        # Apply the filter, passing in the state (zi) from the previous frame
        frame_filtered, zi = sig.lfilter(b, a, frame_amped, zi=zi)

        # Write to output array
        output_audio[i:i+frame_size] = frame_filtered

    # 4. Convolution Reverb (The Environment)
    if ir_file:
        # Load an Impulse Response (e.g., a WAV file of a balloon popping in an open field)
        ir_sr, ir = wav.read(ir_file)
        if ir.dtype == np.int16:
            ir = ir.astype(np.float64) / 32768.0
        if len(ir.shape) > 1:
            ir = ir.mean(axis=1)

        # Use FFT-based convolution to apply the acoustic space to our audio.
        # This is where the Fast Fourier Transform mathematically shines.
        wet_signal = sig.fftconvolve(output_audio, ir, mode='full')

        # Trim the tail to match original roughly, or let it ring out
        wet_signal = wet_signal[:total_samples] 

        # Mix the "Dry" (direct) and "Wet" (reverb) signals. 
        # For a distant sound, it should be mostly wet.
        output_audio = (0.2 * output_audio) + (0.8 * wet_signal)

    # 5. Prevent Clipping and Save
    max_val = np.max(np.abs(output_audio))
    if max_val > 1.0:
        output_audio = output_audio / max_val

    # Convert back to 16-bit PCM
    output_audio_int16 = np.int16(output_audio * 32767.0)
    wav.write(output_file, sr, output_audio_int16)
    print(f"Processing complete. Saved to {output_file}")

# Example execution:
# simulate_wind_and_distance("vocals.wav", "distant_vocals.wav", ir_file="open_field_ir.wav")
def main():
    print("Simulating distant...")
    simulate_wind_and_distance("flower-duet.wav", "distant-flower-duet.wav")
    print("Simulating distant w/IR...")
    simulate_wind_and_distance("flower-duet.wav",
                               "ir-distant-flower-duet.wav",
                               ir_file="ir/selected-ir.wav")

if __name__ == "__main__":
    main()
