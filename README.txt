# Fourier Noise Reduction Application

## Overview

This application provides audio noise reduction capabilities in two modes:

1. File-based processing: Cleans noise from audio files
2. Real-time processing: Reduces noise from microphone input in real-time

## Features

### File Processing Mode

- Reads WAV audio files
- Performs detailed audio analysis
- Applies noise reduction using Fourier transformation
- Generates comprehensive analysis reports
- Saves cleaned audio to a new file

### Real-time Processing Mode

- Captures audio from microphone
- Applies noise reduction in real-time
- Outputs cleaned audio to speakers
- Uses moving average filter for noise reduction

## Technical Details

### Audio Processing Parameters

- Sample Rate: 44.1 kHz
- Channels: Mono (1 channel)
- Frame Size: 1024 samples
- Processing Window: 5 samples (for real-time mode)

### Noise Reduction Algorithm

1. File Mode:

   - Uses FFT (Fast Fourier Transform) for frequency analysis
   - Applies adaptive thresholding
   - Preserves harmonic frequencies
   - Enhances music signals

2. Real-time Mode:
   - Uses moving average filter
   - Applies threshold-based noise reduction
   - Enhances signals above threshold

## Usage

### Command Line Arguments

```bash
go run main.go [options]
```

Options:

- `-input`: Input audio file (WAV format) - Required for file mode
- `-output`: Output audio file (cleaned) - Optional, defaults to input_cleaned.wav
- `-threshold`: Noise filtering threshold value (default: 200.0)
- `-verbose`: Show detailed process information
- `-streamline`: Enable real-time microphone noise reduction

### Examples

1. Process an audio file:

```bash
go run main.go -input=input.wav -output=cleaned.wav -verbose=true
```

2. Start real-time noise reduction:

```bash
go run main.go -streamline
```

## Output

### File Processing Mode

- Creates cleaned audio file
- Generates two analysis reports:
  1. Pre-cleaning analysis
  2. Post-cleaning analysis
- Reports include:
  - Audio quality metrics
  - Spectral analysis
  - Noise analysis
  - Frequency distribution
  - Cleaning parameters

### Real-time Mode

- Processes audio in real-time
- Shows status messages
- Press Ctrl+C to exit

## Dependencies

- github.com/go-audio/audio
- github.com/go-audio/wav
- github.com/gordonklaus/portaudio
- gonum.org/v1/gonum/dsp/fourier

## Requirements

- Go 1.21 or later
- PortAudio library installed on your system
- WAV audio files for file processing mode
- Microphone and speakers for real-time mode

## Notes

- The application automatically adjusts parameters based on audio analysis
- Real-time mode may introduce some latency due to processing
- For best results, use high-quality audio input
- Analysis reports are saved in the ./analyses directory
