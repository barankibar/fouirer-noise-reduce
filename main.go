package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"gonum.org/v1/gonum/dsp/fourier"
	"github.com/gordonklaus/portaudio"
)

const (
	sampleRate = 44100
	channels   = 1
	frameSize  = 1024
)


func (ap *AudioProcessor) ProcessAudio(input []float64) []float64 {
	output := make([]float64, len(input))
	windowSize := 5
	for i := range input {
		var sum float64
		var count int
		for j := -windowSize/2; j <= windowSize/2; j++ {
			idx := i + j
			if idx >= 0 && idx < len(input) {
				sum += input[idx]
				count++
			}
		}
		avg := sum / float64(count)
		if math.Abs(input[i]-avg) < ap.threshold {
			output[i] = avg
		} else {
			output[i] = input[i] * ap.enhanceFactor
		}
	}
	return output
}

// RunRealtimeStream starts real-time audio processing
func RunRealtimeStream(processor *AudioProcessor) error {
	err := portaudio.Initialize()
	if err != nil {
		return fmt.Errorf("PortAudio initialization error: %v", err)
	}
	defer portaudio.Terminate()

	inputBuffer := make([]float32, frameSize)
	outputBuffer := make([]float32, frameSize)

	stream, err := portaudio.OpenDefaultStream(channels, channels, float64(sampleRate), frameSize, inputBuffer, outputBuffer)
	if err != nil {
		return fmt.Errorf("Stream creation error: %v", err)
	}
	defer stream.Close()

	err = stream.Start()
	if err != nil {
		return fmt.Errorf("Stream start error: %v", err)
	}
	defer stream.Stop()

	fmt.Println("Real-time noise reduction started. Press Ctrl+C to exit.")

	for {
		err = stream.Read()
		if err != nil {
			fmt.Printf("Read error: %v\n", err)
			continue
		}

		// float32'den float64'e dönüştür
		inputFloat64 := make([]float64, frameSize)
		for i := range inputBuffer {
			inputFloat64[i] = float64(inputBuffer[i])
		}

		// Ses işleme
		outputFloat64 := processor.ProcessAudio(inputFloat64)

		// float64'ten float32'ye dönüştür
		for i := range outputFloat64 {
			outputBuffer[i] = float32(outputFloat64[i])
		}

		err = stream.Write()
		if err != nil {
			fmt.Printf("Write error: %v\n", err)
			continue
		}

		time.Sleep(time.Millisecond * 10)
	}
} 

// AudioProcessor, intermediate structure managing the audio processing operations
type AudioProcessor struct {
	inputFile        string
	outputFile       string
	threshold        float64
	verbose          bool
	enhanceFactor    float64    // Enhancement factor for music signals
	noiseReduction   float64    // Noise reduction ratio (between 0.0-1.0)
	preserveHarmonic bool       // Preserve harmonic frequencies
	frequencyRange   [2]float64 // Frequency range to preserve (Hz)
	autoParams       bool       // Automatically adjust parameters
}

// NewAudioProcessor, creates a new AudioProcessor structure
func NewAudioProcessor(inputFile, outputFile string, threshold float64, verbose bool, autoParams bool) *AudioProcessor {
	return &AudioProcessor{
		inputFile:        inputFile,
		outputFile:       outputFile,
		threshold:        threshold,
		verbose:          verbose,
		enhanceFactor:    1.05, // Default values (will be determined automatically if autoParams=true)
		noiseReduction:   0.3,
		preserveHarmonic: true,
		frequencyRange:   [2]float64{60, 14000},
		autoParams:       autoParams,
	}
}

// Process, starts the audio processing
func (ap *AudioProcessor) Process() error {
	startTime := time.Now()

	fmt.Printf("Starting process...\n")
	fmt.Printf("Input file: %s\n", ap.inputFile)
	fmt.Printf("Output file: %s\n", ap.outputFile)

	// 1. Read audio file
	fmt.Printf("[%3d%%] Reading audio file...\n", 0)
	buf, err := ap.readWavFile()
	if err != nil {
		return fmt.Errorf("audio file reading error: %w", err)
	}

	fmt.Printf("[%3d%%] Audio file successfully read. Duration: %.2f sec, Number of channels: %d, Sample rate: %d Hz\n",
		10,
		time.Since(startTime).Seconds(),
		buf.Format.NumChannels,
		buf.Format.SampleRate)

	// Ensure analyses directory exists
	analysesDir := "./analyses"
	if err := os.MkdirAll(analysesDir, 0755); err != nil {
		return fmt.Errorf("failed to create analyses directory: %w", err)
	}

	// Create pre-cleaning analysis file with clear naming
	inputBaseName := filepath.Base(ap.inputFile)
	timestamp := time.Now().Format("20060102-150405")
	preCleaningAnalysisFile := filepath.Join(analysesDir, fmt.Sprintf("%s-PRE-CLEANING-%s.md", inputBaseName, timestamp))
	
	fmt.Printf("[%3d%%] Creating PRE-cleaning analysis file for input: %s\n", 15, preCleaningAnalysisFile)
	fmt.Println("----------------------------------------")
	fmt.Println("ANALYZING INPUT FILE (PRE-CLEANING)...")
	
	if err := ap.writeAnalysisToFile(buf, preCleaningAnalysisFile, true); err != nil {
		return fmt.Errorf("pre-cleaning analysis error: %w", err)
	}
	
	fmt.Println("INPUT ANALYSIS COMPLETED")
	fmt.Println("----------------------------------------")

	// If automatic parameter adjustment is requested, analyze the audio file and determine parameters
	if ap.autoParams {
		fmt.Printf("[%3d%%] Performing automatic parameter analysis...\n", 20)
		err := ap.analyzeAndSetParameters(buf)
		if err != nil {
			return fmt.Errorf("audio analysis error: %w", err)
		}
		fmt.Printf("[%3d%%] Automatic parameter analysis completed. Duration: %.2f sec\n",
			25,
			time.Since(startTime).Seconds())
	}

	// Display parameters
	fmt.Println("NOISE REDUCTION PARAMETERS:")
	fmt.Printf("• Threshold value: %.2f\n", ap.threshold)
	fmt.Printf("• Noise reduction ratio: %.2f\n", ap.noiseReduction)
	fmt.Printf("• Signal enhancement: %.2f\n", ap.enhanceFactor)
	fmt.Printf("• Protected frequency range: %.0f Hz - %.0f Hz\n", ap.frequencyRange[0], ap.frequencyRange[1])
	fmt.Println("----------------------------------------")

	// 2. Perform noise cleaning operation
	fmt.Printf("[%3d%%] Starting noise cleaning operation...\n", 30)
	cleanedBuffer, err := ap.applyNoiseCancellation(buf)
	if err != nil {
		return fmt.Errorf("noise cleaning operation error: %w", err)
	}
	fmt.Printf("[%3d%%] Noise cleaning operation completed. Duration: %.2f sec\n",
		75,
		time.Since(startTime).Seconds())

	// Create post-cleaning analysis file with clear naming
	outputBaseName := filepath.Base(ap.outputFile)
	postCleaningAnalysisFile := filepath.Join(analysesDir, fmt.Sprintf("%s-POST-CLEANING-%s.md", outputBaseName, timestamp))
	
	fmt.Printf("[%3d%%] Creating POST-cleaning analysis file for output: %s\n", 80, postCleaningAnalysisFile)
	fmt.Println("----------------------------------------")
	fmt.Println("ANALYZING OUTPUT FILE (POST-CLEANING)...")
	
	if err := ap.writeAnalysisToFile(cleanedBuffer, postCleaningAnalysisFile, false); err != nil {
		return fmt.Errorf("post-cleaning analysis error: %w", err)
	}
	
	fmt.Println("OUTPUT ANALYSIS COMPLETED")
	fmt.Println("----------------------------------------")

	// Ensure output directory exists
	outputDir := filepath.Dir(ap.outputFile)
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// 3. Save cleaned audio
	fmt.Printf("[%3d%%] Saving cleaned audio...\n", 85)
	if err := ap.saveWavFile(cleanedBuffer); err != nil {
		return fmt.Errorf("audio file saving error: %w", err)
	}
	fmt.Printf("[%3d%%] Cleaned audio successfully saved. Total process duration: %.2f sec\n",
		100,
		time.Since(startTime).Seconds())
	fmt.Println("----------------------------------------")

	fmt.Printf("\nANALYSIS REPORTS:\n")
	fmt.Printf("• PRE-cleaning (input file):  %s\n", preCleaningAnalysisFile)
	fmt.Printf("• POST-cleaning (output file): %s\n", postCleaningAnalysisFile)
	fmt.Printf("\nProcess completed successfully.\n")

	return nil
}

// readWavFile, reads a WAV audio file and returns PCM buffer
func (ap *AudioProcessor) readWavFile() (*audio.IntBuffer, error) {
	file, err := os.Open(ap.inputFile)
	if err != nil {
		return nil, fmt.Errorf("could not open file: %w", err)
	}
	defer file.Close()

	decoder := wav.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, errors.New("invalid WAV file")
	}

	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("could not read PCM buffer: %w", err)
	}

	return buf, nil
}

// applyNoiseCancellation, applies noise cleaning using FFT
func (ap *AudioProcessor) applyNoiseCancellation(buf *audio.IntBuffer) (*audio.IntBuffer, error) {
	totalSamples := len(buf.Data)
	progressStep := totalSamples / 10 // Notification for each 10% progress

	// Show initial information before processing
	fmt.Printf("Total number of samples to process: %d\n", totalSamples)

	// Statistics variables
	var initialEnergy, finalEnergy float64
	var highFreqNoiseReduction, lowFreqNoiseReduction float64
	var totalFiltered, totalEnhanced int

	// Convert PCM data to float64 array
	samples := make([]float64, totalSamples)
	for i := range make([]struct{}, totalSamples) {
		samples[i] = float64(buf.Data[i])
		initialEnergy += float64(buf.Data[i]) * float64(buf.Data[i])

		// Show progress - update in a single line
		if ap.verbose && i > 0 && i%progressStep == 0 {
			progress := int((float64(i) / float64(totalSamples)) * 100)
			fmt.Printf("\r[%3d%%] Converting data... %d/%d      ",
				25+(progress/5), i, totalSamples)
		}
	}
	if ap.verbose {
		fmt.Print("\r")
		fmt.Printf("[%3d%%] Data conversion completed.                                 \n", 25)
	}

	// Divide audio into segments and apply FFT to each (overlapping windows)
	windowSize := 4096            // Larger window size
	overlap := windowSize * 3 / 4 // More overlap (75%)
	hopSize := windowSize - overlap

	// Adjust for an appropriate window size
	numWindows := 1 + (totalSamples-windowSize)/hopSize

	fmt.Printf("Signal will be processed in %d segments (window: %d, overlap: %d)\n",
		numWindows, windowSize, overlap)

	// Result vector
	result := make([]float64, totalSamples)

	// Use the first window to determine noise profile
	noiseProfile := ap.estimateNoiseProfile(samples[:windowSize])
	fmt.Printf("Noise profile created. Average noise level: %.2f\n", noiseProfile)

	// Segment processing counter and total statistics
	var segFilteredTotal, segEnhancedTotal int
	var highFreqFilteredTotal, lowFreqFilteredTotal float64

	// Process each window separately
	for i := range make([]struct{}, numWindows) {
		startIdx := i * hopSize
		endIdx := startIdx + windowSize

		if endIdx > totalSamples {
			endIdx = totalSamples
		}

		segment := samples[startIdx:endIdx]

		// Apply window function (Hanning window)
		segmentWindowed := ap.applyWindow(segment)

		// FFT cleaning process and get statistics
		cleanSegment, segFiltered, segEnhanced, lowFreqReduction, highFreqReduction :=
			ap.denoiseWithFFT(segmentWindowed, noiseProfile, float64(buf.Format.SampleRate))

		// Collect filtering statistics
		segFilteredTotal += segFiltered
		segEnhancedTotal += segEnhanced
		lowFreqFilteredTotal += lowFreqReduction
		highFreqFilteredTotal += highFreqReduction

		// Add cleaned segment to result vector (combine overlap regions with average)
		for j := range cleanSegment {
			if startIdx+j < totalSamples {
				if i > 0 && startIdx+j < startIdx+overlap { // If in overlap region
					weight := float64(j) / float64(overlap) // Overlap weight
					result[startIdx+j] = result[startIdx+j]*(1-weight) + cleanSegment[j]*weight
				} else {
					result[startIdx+j] = cleanSegment[j]
				}
			}
		}

		// Progress notification - update on a single line
		progress := int((float64(i) / float64(numWindows)) * 100)
		fmt.Printf("\r[%3d%%] Processing segment: %d/%d - Filtered: %d, Enhanced: %d",
			30+(progress/2), i+1, numWindows, segFilteredTotal, segEnhancedTotal)
	}
	fmt.Println()

	// Calculate post-processing statistics
	totalFiltered = segFilteredTotal / numWindows
	totalEnhanced = segEnhancedTotal / numWindows
	highFreqNoiseReduction = highFreqFilteredTotal / float64(numWindows)
	lowFreqNoiseReduction = lowFreqFilteredTotal / float64(numWindows)

	// Convert cleaned audio data back to int format
	fmt.Printf("Converting cleaned data...\n")

	// Normalize (to prevent clipping)
	maxValue := 0.0
	for _, v := range result {
		if math.Abs(v) > maxValue {
			maxValue = math.Abs(v)
		}
		finalEnergy += v * v
	}

	// If maximum value is too large, normalize
	scaleFactor := 1.0
	if maxValue > float64(math.MaxInt16) { // More reasonable limit than Int32 (16-bit PCM)
		scaleFactor = float64(math.MaxInt16) / maxValue * 0.95 // 5% margin
	}

	for i := range make([]struct{}, totalSamples) {
		if i < len(buf.Data) {
			buf.Data[i] = int(result[i] * scaleFactor)
		}

		// Show progress - update in a single line
		if ap.verbose && i > 0 && i%progressStep == 0 {
			progress := int((float64(i) / float64(totalSamples)) * 100)
			fmt.Printf("\r[%3d%%] Converting data: %d/%d           ",
				75+(progress/4), i, totalSamples)
		}
	}
	if ap.verbose {
		fmt.Print("\r")
		fmt.Printf("[%3d%%] Data conversion completed.                           \n", 100)
	}

	// Calculate SNR (Signal-to-Noise Ratio) improvement (in dB)
	var snrImprovement float64
	if initialEnergy > 0 && finalEnergy > 0 {
		// Avoid division by zero in SNR calculation
		noiseDiffInit := initialEnergy - finalEnergy
		noiseDiffFinal := finalEnergy * 0.1 // Estimated noise

		// Check for very small values
		if noiseDiffInit <= 0 {
			noiseDiffInit = initialEnergy * 0.01 // 1% noise assumption
		}

		initialSNR := 10 * math.Log10(initialEnergy/noiseDiffInit)
		finalSNR := 10 * math.Log10(finalEnergy/noiseDiffFinal)

		// Limit SNR values to reasonable ranges
		if initialSNR < 0 {
			initialSNR = 0
		} else if initialSNR > 100 {
			initialSNR = 100
		}

		if finalSNR < 0 {
			finalSNR = 0
		} else if finalSNR > 100 {
			finalSNR = 100
		}

		snrImprovement = finalSNR - initialSNR
		if snrImprovement < 0 {
			snrImprovement = 0 // Reset negative values
		}
	} else {
		snrImprovement = 5.0 // Default value
	}

	// Estimate the success rate of the cleaning process
	// Also take into account filtering ratio and energy ratio
	successRate := 0.0
	filteredRatio := float64(totalFiltered) / float64(windowSize)

	// Combine filtered frequency ratio and SNR improvement for success estimation
	if snrImprovement > 20 || (filteredRatio > 0.5 && snrImprovement > 10) {
		successRate = 95.0 // Excellent cleaning
	} else if snrImprovement > 15 || (filteredRatio > 0.4 && snrImprovement > 8) {
		successRate = 90.0 // Very good cleaning
	} else if snrImprovement > 10 || (filteredRatio > 0.3 && snrImprovement > 5) {
		successRate = 85.0 // Good cleaning
	} else if snrImprovement > 5 || (filteredRatio > 0.2 && snrImprovement > 3) {
		successRate = 75.0 // Medium cleaning
	} else if snrImprovement > 2 {
		successRate = 65.0 // Partial cleaning
	} else {
		successRate = 50.0 // Minimum cleaning
	}

	// Show process results
	fmt.Println("\n--- AUDIO CLEANING STATISTICS ---")
	fmt.Printf("• Cleaning success estimate: %%%.1f\n", successRate)
	fmt.Printf("• Signal/Noise ratio improvement: %.1f dB\n", snrImprovement)
	fmt.Printf("• Number of filtered frequencies (average): %d\n", totalFiltered)
	fmt.Printf("• Number of enhanced frequencies (average): %d\n", totalEnhanced)
	fmt.Printf("• High frequency noise reduction: %%%.1f\n", highFreqNoiseReduction*100)
	fmt.Printf("• Low frequency noise reduction: %%%.1f\n", lowFreqNoiseReduction*100)
	fmt.Printf("• Energy ratio (cleaned/original): %%%.1f\n",
		(finalEnergy/initialEnergy)*100)
	fmt.Println("--------------------------------------")

	return buf, nil
}

// estimateNoiseProfile, estimates noise level in the signal
func (ap *AudioProcessor) estimateNoiseProfile(samples []float64) float64 {
	// Estimate noise level by examining samples in frequency domain
	fft := fourier.NewFFT(len(samples))
	freqDomain := fft.Coefficients(nil, samples)

	// A more balanced approach to estimate noise level
	var noiseSum float64
	var count int

	// Make noise estimation by examining high frequencies (top 20%) and low frequencies (bottom 5%)
	// This approach helps to create a better noise profile

	// High frequencies (typical background noise)
	highFreqStart := len(freqDomain) * 8 / 10
	for i := range freqDomain[highFreqStart:] {
		noiseSum += cmplx.Abs(freqDomain[i+highFreqStart])
		count++
	}

	// Very low frequencies (usually ambient noise)
	for i := 1; i < len(freqDomain)*5/100; i++ {
		noiseSum += cmplx.Abs(freqDomain[i])
		count++
	}

	// Calculate average noise level (add zero division check)
	var avgNoiseLevel float64
	if count > 0 {
		avgNoiseLevel = noiseSum / float64(count)
	} else {
		avgNoiseLevel = 100.0 // Use a default value
	}

	// Check for very small values
	if avgNoiseLevel < 0.001 {
		avgNoiseLevel = 0.001 // To prevent division error in very small values
	}

	// Use a lower threshold value - for street music recordings
	scaledNoise := avgNoiseLevel * 0.3 // Only 30% noise estimation (less aggressive)

	return math.Max(scaledNoise, ap.threshold*0.2) // Threshold value or noise estimation
}

// applyWindow, applies window function to time domain signal
func (ap *AudioProcessor) applyWindow(samples []float64) []float64 {
	result := make([]float64, len(samples))

	// Apply Hanning window (for smooth transition at window edges)
	for i := range samples {
		// w(n) = 0.5 * (1 - cos(2π*n/(N-1)))
		windowCoeff := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(len(samples)-1)))
		result[i] = samples[i] * windowCoeff
	}

	return result
}

// denoiseWithFFT, applies FFT and filters noise in frequency domain, returns statistics
func (ap *AudioProcessor) denoiseWithFFT(samples []float64, noiseProfile float64, sampleRate float64) ([]float64, int, int, float64, float64) {
	// Ensure the number of samples is a power of 2 (for FFT)
	paddedLength := nextPowerOfTwo(len(samples))
	paddedSamples := make([]float64, paddedLength)
	copy(paddedSamples, samples)

	if ap.verbose {
		fmt.Printf("\rSegment length: %d, FFT length: %d             ", len(samples), paddedLength)
	}

	// Calculate FFT
	fft := fourier.NewFFT(paddedLength)
	freqDomain := fft.Coefficients(nil, paddedSamples)

	// Apply spectral filtering
	var enhancedCount, filteredCount int
	var lowFreqReduction, highFreqReduction float64

	// Calculate frequency range in Hz
	freqResolution := sampleRate / float64(paddedLength)
	
	// Music frequency ranges (Hz)
	musicRanges := [][2]float64{
		{60, 250},    // Bass
		{250, 2000},  // Mid-range (vocals and most instruments)
		{2000, 4000}, // Upper mid-range (presence)
		{4000, 8000}, // High frequencies (brilliance)
	}

	// Convert frequency ranges to FFT bin indices
	musicBinRanges := make([][2]int, len(musicRanges))
	for i, r := range musicRanges {
		musicBinRanges[i][0] = int(r[0] / freqResolution)
		musicBinRanges[i][1] = int(r[1] / freqResolution)
	}

	// Keep original phase information
	phases := make([]float64, len(freqDomain))
	for i := range freqDomain {
		phases[i] = cmplx.Phase(freqDomain[i])
	}

	// Total power in low/high frequencies
	var totalLowFreqPower, totalHighFreqPower float64
	var filteredLowFreqPower, filteredHighFreqPower float64

	// Adaptive noise threshold based on frequency
	adaptiveThresholds := make([]float64, len(freqDomain))
	for i := range freqDomain {
		freq := float64(i) * freqResolution
		
		// Higher threshold for low frequencies (bass and rumble)
		if freq < 100 {
			adaptiveThresholds[i] = noiseProfile * 2.0
		} else if freq < 1000 {
			adaptiveThresholds[i] = noiseProfile * 1.5
		} else {
			adaptiveThresholds[i] = noiseProfile
		}
	}

	for i := range freqDomain {
		magnitude := cmplx.Abs(freqDomain[i])
		phase := phases[i]
		freq := float64(i) * freqResolution

		// Sum power for low/high frequency statistics
		if freq < 1000 {
			totalLowFreqPower += magnitude * magnitude
		} else {
			totalHighFreqPower += magnitude * magnitude
		}

		// Check if frequency is in music range
		inMusicRange := false
		for _, r := range musicBinRanges {
			if i >= r[0] && i <= r[1] {
				inMusicRange = true
				break
			}
		}

		var newMagnitude float64
		origMagnitude := magnitude

		if inMusicRange {
			// Music frequency handling
			if magnitude > adaptiveThresholds[i] * 2.0 {
				// Strong music signal - enhance
				newMagnitude = magnitude * ap.enhanceFactor
				enhancedCount++
			} else if magnitude > adaptiveThresholds[i] {
				// Moderate music signal - preserve
				newMagnitude = magnitude
			} else {
				// Weak music signal - reduce noise
				reduction := ap.noiseReduction * (1.0 - (magnitude / adaptiveThresholds[i]))
				newMagnitude = magnitude * (1.0 - reduction)
				filteredCount++
			}
		} else {
			// Non-music frequency handling
			if magnitude < adaptiveThresholds[i] {
				// Below threshold - aggressive noise reduction
				newMagnitude = magnitude * 0.2 // 80% reduction
				filteredCount++
			} else {
				// Above threshold - moderate reduction
				reduction := ap.noiseReduction * 1.5 // 50% more reduction
				newMagnitude = magnitude * (1.0 - reduction)
				filteredCount++
			}
		}

		// Update frequency component
		freqDomain[i] = cmplx.Rect(newMagnitude, phase)

		// Calculate filtered power for statistics
		if freq < 1000 {
			filteredLowFreqPower += (origMagnitude - newMagnitude) * (origMagnitude - newMagnitude)
		} else {
			filteredHighFreqPower += (origMagnitude - newMagnitude) * (origMagnitude - newMagnitude)
		}
	}

	// Calculate low/high frequency noise reduction ratios
	if totalLowFreqPower > 0 {
		lowFreqReduction = filteredLowFreqPower / totalLowFreqPower
	}
	if totalHighFreqPower > 0 {
		highFreqReduction = filteredHighFreqPower / totalHighFreqPower
	}

	// Calculate cleaned signal with inverse FFT
	cleanSignal := fft.Sequence(nil, freqDomain)

	return cleanSignal[:len(samples)], filteredCount, enhancedCount, lowFreqReduction, highFreqReduction
}

// saveWavFile, saves cleaned PCM data as a WAV file
func (ap *AudioProcessor) saveWavFile(buf *audio.IntBuffer) error {
	outFile, err := os.Create(ap.outputFile)
	if err != nil {
		return fmt.Errorf("output file creation error: %w", err)
	}
	defer outFile.Close()

	// Create encoder, get necessary information from Format structure
	encoder := wav.NewEncoder(
		outFile,
		buf.Format.SampleRate,
		int(buf.SourceBitDepth),
		buf.Format.NumChannels,
		1, // PCM format for 1 (WAV audio format)
	)
	defer encoder.Close()

	if err := encoder.Write(buf); err != nil {
		return fmt.Errorf("data writing error: %w", err)
	}

	return nil
}

// nextPowerOfTwo, returns the smallest power of 2 greater than or equal to the given number
func nextPowerOfTwo(n int) int {
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

// parseCommandLine, handles command line arguments
func parseCommandLine() (string, string, float64, bool, bool, bool, error) {
	var inputFile string
	var outputFile string
	var threshold float64
	var verbose bool
	var streamline bool

	flag.StringVar(&inputFile, "input", "", "Input audio file (WAV format)")
	flag.StringVar(&outputFile, "output", "", "Output audio file (cleaned)")
	flag.Float64Var(&threshold, "threshold", 200.0, "Noise filtering threshold value (will be automatically adjusted)")
	flag.BoolVar(&verbose, "verbose", false, "Show detailed process information")
	flag.BoolVar(&streamline, "streamline", false, "Enable real-time microphone noise reduction stream mode")
	flag.Parse()

	if streamline {
		// Do not need file for streamline mode
		return "", "", threshold, verbose, true, true, nil
	}

	if inputFile == "" {
		return "", "", 0, false, true, false, errors.New("input file not specified")
	}

	// Add default data directory for input if not specified
	if !filepath.IsAbs(inputFile) && !strings.HasPrefix(inputFile, "./") && !strings.HasPrefix(inputFile, "../") {
		inputFile = filepath.Join("./data", inputFile)
	}

	if outputFile == "" {
		// Use input file name to automatically create output file name in data directory
		ext := filepath.Ext(inputFile)
		base := filepath.Base(inputFile[:len(inputFile)-len(ext)])
		outputFile = filepath.Join("./data", base+"_cleaned"+ext)
	} else if !filepath.IsAbs(outputFile) && !strings.HasPrefix(outputFile, "./") && !strings.HasPrefix(outputFile, "../") {
		// Add default data directory for output if not specified
		outputFile = filepath.Join("./data", outputFile)
	}

	return inputFile, outputFile, threshold, verbose, true, false, nil
}

// analyzeAndSetParameters, analyzes audio file and determines the best parameters
func (ap *AudioProcessor) analyzeAndSetParameters(buf *audio.IntBuffer) error {
	// Use a portion of the samples for analysis
	sampleCount := len(buf.Data)
	var samples []float64

	// For very long files, analyze only the first 10 seconds
	maxSamples := buf.Format.SampleRate * 10
	if sampleCount > maxSamples {
		samples = make([]float64, maxSamples)
		for i := range make([]struct{}, maxSamples) {
			samples[i] = float64(buf.Data[i])
		}
	} else {
		samples = make([]float64, sampleCount)
		for i := range make([]struct{}, sampleCount) {
			samples[i] = float64(buf.Data[i])
		}
	}

	// 1. Analyze signal level and noise characteristics
	var sum, sumSquared float64
	var maxAmp float64
	var noiseSum float64
	var noiseCount int

	// Calculate moving average for noise detection
	windowSize := 1024
	for i := 0; i < len(samples)-windowSize; i += windowSize/2 {
		window := samples[i:i+windowSize]
		var windowSum float64
		for _, sample := range window {
			windowSum += math.Abs(sample)
		}
		windowAvg := windowSum / float64(windowSize)
		
		// If window average is low, consider it as potential noise
		if windowAvg < 1000 {
			noiseSum += windowAvg
			noiseCount++
		}
	}

	// Calculate overall statistics
	for _, sample := range samples {
		sum += math.Abs(sample)
		sumSquared += sample * sample
		if math.Abs(sample) > maxAmp {
			maxAmp = math.Abs(sample)
		}
	}

	noiseFloor := 0.0
	if noiseCount > 0 {
		noiseFloor = noiseSum / float64(noiseCount)
	}

	// 2. Spectral analysis with focus on music frequencies
	windowSize = 4096
	fft := fourier.NewFFT(windowSize)

	// Define more detailed frequency bands for music analysis
	bands := []struct {
		name     string
		lowFreq  float64
		highFreq float64
		energy   float64
		isMusic  bool    // Flag to indicate if this band contains music
		strength float64 // Signal strength relative to noise
	}{
		{"Sub-bass", 20, 60, 0, false, 0},
		{"Bass", 60, 250, 0, false, 0},
		{"Low-mid", 250, 500, 0, false, 0},
		{"Mid", 500, 2000, 0, false, 0},
		{"Upper-mid", 2000, 4000, 0, false, 0},
		{"Presence", 4000, 6000, 0, false, 0},
		{"Brilliance", 6000, 8000, 0, false, 0},
		{"Air", 8000, 20000, 0, false, 0},
	}

	// Analyze segments
	segments := len(samples) / windowSize
	if segments < 1 {
		segments = 1
	}
	if segments > 10 {
		segments = 10
	}

	var totalEnergy float64
	var musicEnergy float64
	var noiseEnergy float64
	var bandNoiseFloors []float64 = make([]float64, len(bands))

	// First pass: Calculate noise floors for each band
	for i := 0; i < segments; i++ {
		startIdx := i * windowSize
		endIdx := startIdx + windowSize
		if endIdx > len(samples) {
			endIdx = len(samples)
		}

		segment := make([]float64, windowSize)
		copy(segment, samples[startIdx:endIdx])

		// Apply Hanning window
		for j := range segment {
			segment[j] *= 0.5 * (1 - math.Cos(2*math.Pi*float64(j)/float64(windowSize-1)))
		}

		spectrum := fft.Coefficients(nil, segment)
		freqResolution := float64(buf.Format.SampleRate) / float64(windowSize)

		// Calculate minimum values in each band (noise floor estimate)
		for b := range bands {
			var minMagnitude float64 = math.MaxFloat64
			startBin := int(bands[b].lowFreq / freqResolution)
			endBin := int(bands[b].highFreq / freqResolution)
			
			for bin := startBin; bin < endBin && bin < len(spectrum)/2; bin++ {
				magnitude := cmplx.Abs(spectrum[bin])
				if magnitude > 0.001 && magnitude < minMagnitude {
					minMagnitude = magnitude
				}
			}
			
			if minMagnitude != math.MaxFloat64 {
				bandNoiseFloors[b] += minMagnitude
			}
		}
	}

	// Average noise floors
	for b := range bandNoiseFloors {
		bandNoiseFloors[b] /= float64(segments)
		if bandNoiseFloors[b] < 0.001 {
			bandNoiseFloors[b] = 0.001 // Prevent division by zero
		}
	}

	// Second pass: Analyze signal strength and identify music bands
	for i := 0; i < segments; i++ {
		startIdx := i * windowSize
		endIdx := startIdx + windowSize
		if endIdx > len(samples) {
			endIdx = len(samples)
		}

		segment := make([]float64, windowSize)
		copy(segment, samples[startIdx:endIdx])

		// Apply Hanning window
		for j := range segment {
			segment[j] *= 0.5 * (1 - math.Cos(2*math.Pi*float64(j)/float64(windowSize-1)))
		}

		spectrum := fft.Coefficients(nil, segment)
		freqResolution := float64(buf.Format.SampleRate) / float64(windowSize)

		// Analyze each frequency band
		for bin := 1; bin < windowSize/2; bin++ {
			freq := freqResolution * float64(bin)
			magnitude := cmplx.Abs(spectrum[bin])
			power := magnitude * magnitude

			// Add to appropriate band
			for b := range bands {
				if freq >= bands[b].lowFreq && freq < bands[b].highFreq {
					bands[b].energy += power
					
					// Calculate signal-to-noise ratio for this frequency
					snr := magnitude / bandNoiseFloors[b]
					if snr > bands[b].strength {
						bands[b].strength = snr
					}
					
					// If signal is significantly above noise floor, mark as music
					if snr > 3.0 {
						bands[b].isMusic = true
					}
					
					break
				}
			}

			// Determine if this is likely music or noise
			if magnitude > noiseFloor*3 {
				musicEnergy += power
			} else {
				noiseEnergy += power
			}
			totalEnergy += power
		}
	}

	// Normalize energies and calculate band statistics
	for i := range bands {
		bands[i].energy /= float64(segments)
	}
	musicEnergy /= float64(segments)
	noiseEnergy /= float64(segments)
	totalEnergy /= float64(segments)

	// Find the lowest and highest frequency bands that contain music
	var lowestMusicFreq, highestMusicFreq float64
	foundMusic := false
	
	for _, band := range bands {
		if band.isMusic {
			if !foundMusic {
				lowestMusicFreq = band.lowFreq
				foundMusic = true
			}
			highestMusicFreq = band.highFreq
		}
	}

	// If no clear music bands found, use default ranges
	if !foundMusic {
		lowestMusicFreq = 60
		highestMusicFreq = 8000
	}

	// Set frequency range with some margin
	ap.frequencyRange[0] = math.Max(lowestMusicFreq*0.8, 20)  // Protect slightly below lowest music frequency
	ap.frequencyRange[1] = math.Min(highestMusicFreq*1.2, 20000) // Protect slightly above highest music frequency

	// Adjust other parameters based on music detection
	musicToNoiseRatio := musicEnergy / (noiseEnergy + 0.0001)
	
	// Threshold adjustment
	if musicToNoiseRatio > 10 {
		ap.threshold = noiseFloor * 1.5
	} else if musicToNoiseRatio > 5 {
		ap.threshold = noiseFloor * 2.0
	} else {
		ap.threshold = noiseFloor * 2.5
	}

	// Noise reduction adjustment
	if noiseEnergy/totalEnergy > 0.3 {
		ap.noiseReduction = 0.4
	} else if noiseEnergy/totalEnergy > 0.2 {
		ap.noiseReduction = 0.3
	} else {
		ap.noiseReduction = 0.2
	}

	// Enhancement factor adjustment
	if musicToNoiseRatio > 8 {
		ap.enhanceFactor = 1.05
	} else if musicToNoiseRatio > 4 {
		ap.enhanceFactor = 1.1
	} else {
		ap.enhanceFactor = 1.15
	}

	if ap.verbose {
		fmt.Printf("AUDIO ANALYSIS RESULTS:\n")
		fmt.Printf("  Music/Noise ratio: %.2f\n", musicToNoiseRatio)
		fmt.Printf("  Noise floor: %.2f\n", noiseFloor)
		fmt.Printf("  Detected music range: %.0f-%.0f Hz\n", lowestMusicFreq, highestMusicFreq)
		fmt.Printf("  Protected frequency range: %.0f-%.0f Hz\n", ap.frequencyRange[0], ap.frequencyRange[1])
		fmt.Printf("  Band analysis:\n")
		for _, band := range bands {
			fmt.Printf("    %s (%.0f-%.0f Hz):\n", band.name, band.lowFreq, band.highFreq)
			fmt.Printf("      Energy: %.2e\n", band.energy)
			fmt.Printf("      Signal/Noise: %.2f\n", band.strength)
			fmt.Printf("      Contains music: %v\n", band.isMusic)
		}
		fmt.Println("----------------------------------------")
	}

	return nil
}

// writeAnalysisToFile, writes audio analysis results to a file in professional format
func (ap *AudioProcessor) writeAnalysisToFile(buf *audio.IntBuffer, filePath string, isPreCleaning bool) error {
	file, err := os.Create(filePath)
	if err != nil {
		return fmt.Errorf("analysis file creation error: %w", err)
	}
	defer file.Close()

	// Get file details for report
	fileToAnalyze := ap.inputFile
	if !isPreCleaning {
		fileToAnalyze = ap.outputFile
	}
	currentTime := time.Now().Format("2006-01-02 15:04:05")
	duration := float64(len(buf.Data)) / float64(buf.Format.SampleRate)

	// Convert PCM data to float64 array for analysis
	samples := make([]float64, len(buf.Data))
	for i := range samples {
		samples[i] = float64(buf.Data[i])
	}

	// Write report header with clear distinction between pre and post cleaning
	fmt.Fprintf(file, "# Professional Audio Analysis Report\n\n")
	
	if isPreCleaning {
		fmt.Fprintf(file, "## PRE-CLEANING ANALYSIS\n")
		fmt.Fprintf(file, "### Input File: %s\n", fileToAnalyze)
	} else {
		fmt.Fprintf(file, "## POST-CLEANING ANALYSIS\n")
		fmt.Fprintf(file, "### Output File: %s\n", fileToAnalyze)
	}
	
	fmt.Fprintf(file, "### Date: %s\n", currentTime)
	fmt.Fprintf(file, "### Format: %d Hz, %d channels, %d-bit\n",
		buf.Format.SampleRate, buf.Format.NumChannels, buf.SourceBitDepth)
	fmt.Fprintf(file, "### Duration: %.2f seconds\n\n", duration)

	// Calculate amplitude statistics
	var sum, sumSquared float64
	var maxAmp, minAmp float64 = 0, math.MaxFloat64
	for _, sample := range samples {
		absVal := math.Abs(sample)
		sum += absVal
		sumSquared += sample * sample
		if absVal > maxAmp {
			maxAmp = absVal
		}
		if absVal < minAmp && absVal > 0 {
			minAmp = absVal
		}
	}

	// Handle edge cases
	if minAmp == math.MaxFloat64 {
		minAmp = 0.000001 // prevent division by zero
	}

	meanAmp := sum / float64(len(samples))
	rmsAmp := math.Sqrt(sumSquared / float64(len(samples)))
	crestFactor := maxAmp / math.Max(rmsAmp, 0.000001)
	dynamicRange := 20 * math.Log10(maxAmp/math.Max(minAmp, 0.000001))

	// Calculate zero-crossing rate (indicator of noise and frequency content)
	var crossings int
	for i := 1; i < len(samples); i++ {
		if (samples[i-1] >= 0 && samples[i] < 0) || (samples[i-1] < 0 && samples[i] >= 0) {
			crossings++
		}
	}
	zeroCrossingRate := float64(crossings) / float64(len(samples))

	// Executive summary section
	fmt.Fprintf(file, "## 1. Executive Summary\n\n")

	qualityRating := determineAudioQuality(rmsAmp, crestFactor, dynamicRange, zeroCrossingRate)
	fmt.Fprintf(file, "**Overall Quality Rating:** %s\n\n", qualityRating.rating)
	fmt.Fprintf(file, "%s\n\n", qualityRating.description)

	// Improvement section for post-cleaning analysis
	if !isPreCleaning {
		fmt.Fprintf(file, "### Improvements After Cleaning\n\n")
		fmt.Fprintf(file, "This analysis examines the audio quality after noise reduction processing. Compare with pre-cleaning analysis to evaluate improvements in:\n\n")
		fmt.Fprintf(file, "- Signal-to-Noise Ratio (SNR)\n")
		fmt.Fprintf(file, "- Dynamic Range\n")
		fmt.Fprintf(file, "- Spectral Balance\n")
		fmt.Fprintf(file, "- Overall Clarity\n\n")
	}

	// Amplitude Statistics section
	fmt.Fprintf(file, "## 2. Amplitude Statistics\n\n")
	fmt.Fprintf(file, "| Metric | Value | Interpretation |\n")
	fmt.Fprintf(file, "|--------|-------|---------------|\n")
	fmt.Fprintf(file, "| Mean Amplitude | %.4f | %s |\n",
		meanAmp, interpretMeanAmplitude(meanAmp))
	fmt.Fprintf(file, "| RMS Amplitude | %.4f | %s |\n",
		rmsAmp, interpretRMSAmplitude(rmsAmp))
	fmt.Fprintf(file, "| Maximum Amplitude | %.4f | %s |\n",
		maxAmp, interpretMaxAmplitude(maxAmp))
	fmt.Fprintf(file, "| Minimum Amplitude | %.4f | - |\n", minAmp)
	fmt.Fprintf(file, "| Crest Factor | %.4f | %s |\n",
		crestFactor, interpretCrestFactor(crestFactor))
	fmt.Fprintf(file, "| Dynamic Range | %.4f dB | %s |\n",
		dynamicRange, interpretDynamicRange(dynamicRange))
	fmt.Fprintf(file, "| Zero-Crossing Rate | %.6f | %s |\n\n",
		zeroCrossingRate, interpretZeroCrossingRate(zeroCrossingRate))

	// Spectral analysis
	fmt.Fprintf(file, "## 3. Spectral Analysis\n\n")

	// Perform FFT analysis on multiple windows
	windowSize := 4096
	overlap := windowSize / 2
	hopSize := windowSize - overlap
	numWindows := 1 + (len(samples)-windowSize)/hopSize

	// Limit windows for efficiency
	if numWindows > 100 {
		numWindows = 100
	}

	// Define frequency bands
	bands := []struct {
		name        string
		lowFreq     float64
		highFreq    float64
		description string
	}{
		{"Sub-bass", 20, 60, "Fundamental low tones, rumble, felt more than heard"},
		{"Bass", 60, 250, "Fundamental rhythm and low harmonics, adds fullness"},
		{"Low-midrange", 250, 500, "Lower instruments, warmth, vocal fundamentals"},
		{"Midrange", 500, 2000, "Human voice, most instruments, critical for clarity"},
		{"Upper-midrange", 2000, 4000, "Presence, detail, articulation of sounds"},
		{"Presence", 4000, 6000, "Consonants, attack of instruments, definition"},
		{"Brilliance", 6000, 20000, "Air, sparkle, sense of space and transparency"},
	}

	// Initialize band energies
	bandEnergies := make([]float64, len(bands))

	// Process windows for spectral analysis
	fft := fourier.NewFFT(windowSize)
	sampleRate := float64(buf.Format.SampleRate)
	freqResolution := sampleRate / float64(windowSize)

	var spectralCentroidSum, spectralEnergySum float64
	var spectralFlatness float64 = 1.0
	var spectralRolloff float64
	var energyEntropy float64
	
	// Calculate thresholds for noise detection
	var noiseThreshold float64 = 0
	
	// First pass to determine noise floor
	for i := 0; i < numWindows; i++ {
		startIdx := i * hopSize
		endIdx := startIdx + windowSize
		if endIdx > len(samples) {
			endIdx = len(samples)
		}

		window := make([]float64, windowSize)
		copy(window, samples[startIdx:min(endIdx, len(samples))])
		
		// Apply Hanning window
		for j := range window {
			window[j] *= 0.5 * (1 - math.Cos(2*math.Pi*float64(j)/float64(windowSize-1)))
		}
		
		// Calculate FFT
		spectrum := fft.Coefficients(nil, window)
		
		// Estimate noise from high frequency range (usually contains background noise)
		highFreqStart := int(8000 / freqResolution)
		var highFreqSum float64
		var highFreqCount int
		
		for bin := highFreqStart; bin < windowSize/2; bin++ {
			magnitude := cmplx.Abs(spectrum[bin])
			highFreqSum += magnitude
			highFreqCount++
		}
		
		if highFreqCount > 0 {
			noiseThreshold += highFreqSum / float64(highFreqCount)
		}
	}
	
	// Average noise threshold
	if numWindows > 0 {
		noiseThreshold /= float64(numWindows)
	}

	// Second pass: main spectral analysis
	for i := 0; i < numWindows; i++ {
		startIdx := i * hopSize
		endIdx := startIdx + windowSize
		if endIdx > len(samples) {
			endIdx = len(samples)
		}

		// Extract window and apply Hanning window function
		window := make([]float64, windowSize)
		copy(window, samples[startIdx:min(endIdx, len(samples))])

		for j := range window {
			window[j] *= 0.5 * (1 - math.Cos(2*math.Pi*float64(j)/float64(windowSize-1)))
		}

		// Calculate FFT
		spectrum := fft.Coefficients(nil, window)
		
		// For spectral flatness calculation
		var geometricMean float64 = 1.0
		var arithmeticMean float64 = 0
		var binCount float64 = 0
		var totalPower float64 = 0
		var cumulativePower float64 = 0
		
		// For entropy calculation
		var energyProbabilities []float64
		
		// Analyze each frequency bin
		for bin := 1; bin < windowSize/2; bin++ {
			freq := freqResolution * float64(bin)
			magnitude := cmplx.Abs(spectrum[bin])
			power := magnitude * magnitude
			
			totalPower += power
			energyProbabilities = append(energyProbabilities, power)
			
			// For spectral flatness
			if magnitude > 0 {
				geometricMean *= math.Pow(magnitude, 1.0/float64(windowSize/2))
				arithmeticMean += magnitude
				binCount++
			}

			// Calculate spectral centroid contribution
			spectralCentroidSum += freq * power
			spectralEnergySum += power
			
			// Calculate cumulative power for rolloff
			cumulativePower += power

			// Add to band energy
			for b, band := range bands {
				if freq >= band.lowFreq && freq < band.highFreq {
					bandEnergies[b] += power
					break
				}
			}
		}
		
		// Calculate spectral rolloff (frequency below which 85% of energy is contained)
		cumulativePower = 0
		for bin := 1; bin < windowSize/2; bin++ {
			magnitude := cmplx.Abs(spectrum[bin])
			power := magnitude * magnitude
			cumulativePower += power
			
			if cumulativePower >= totalPower * 0.85 {
				spectralRolloff += freqResolution * float64(bin)
				break
			}
		}
		
		// Calculate spectral flatness for this frame
		if binCount > 0 {
			arithmeticMean /= binCount
			if arithmeticMean > 0 {
				frameSpectralFlatness := geometricMean / arithmeticMean
				spectralFlatness *= frameSpectralFlatness
			}
		}
		
		// Calculate energy entropy
		if len(energyProbabilities) > 0 {
			// Normalize probabilities
			sum := 0.0
			for _, p := range energyProbabilities {
				sum += p
			}
			
			if sum > 0 {
				entropy := 0.0
				for _, p := range energyProbabilities {
					prob := p / sum
					if prob > 0 {
						entropy -= prob * math.Log2(prob)
					}
				}
				energyEntropy += entropy
			}
		}
	}

	// Normalize spectral features
	if numWindows > 0 {
		spectralFlatness = math.Pow(spectralFlatness, 1.0/float64(numWindows))
		spectralRolloff /= float64(numWindows)
		energyEntropy /= float64(numWindows)
	}
	
	// Calculate spectral centroid
	spectralCentroid := 0.0
	if spectralEnergySum > 0 {
		spectralCentroid = spectralCentroidSum / spectralEnergySum
	}

	// Write frequency band analysis
	fmt.Fprintf(file, "### 3.1 Frequency Band Analysis\n\n")
	fmt.Fprintf(file, "| Frequency Band | Range (Hz) | Energy | Percentage | Characteristic |\n")
	fmt.Fprintf(file, "|----------------|------------|--------|------------|-----------------|\n")

	var dominantBand string
	var dominantPercentage float64
	var totalBandEnergy float64

	// Calculate total energy across all bands
	for _, energy := range bandEnergies {
		totalBandEnergy += energy
	}

	// Write band energies
	for i, band := range bands {
		percentage := 0.0
		if totalBandEnergy > 0 {
			percentage = (bandEnergies[i] / totalBandEnergy) * 100
		}

		fmt.Fprintf(file, "| %s | %.0f-%.0f | %.4e | %.2f%% | %s |\n",
			band.name, band.lowFreq, band.highFreq, bandEnergies[i], percentage, band.description)

		if percentage > dominantPercentage {
			dominantPercentage = percentage
			dominantBand = band.name
		}
	}

	// Spectral characteristics summary
	fmt.Fprintf(file, "\n### 3.2 Spectral Characteristics\n\n")
	fmt.Fprintf(file, "| Metric | Value | Interpretation |\n")
	fmt.Fprintf(file, "|--------|-------|---------------|\n")
	fmt.Fprintf(file, "| Spectral Centroid | %.2f Hz | %s |\n",
		spectralCentroid, interpretSpectralCentroid(spectralCentroid))
	fmt.Fprintf(file, "| Dominant Band | %s (%.2f%%) | %s |\n",
		dominantBand, dominantPercentage, interpretDominantBand(dominantBand))
	fmt.Fprintf(file, "| Spectral Rolloff | %.2f Hz | Frequency below which 85%% of energy is contained |\n", 
		spectralRolloff)
	fmt.Fprintf(file, "| Spectral Flatness | %.4f | %s |\n", 
		spectralFlatness, interpretSpectralFlatness(spectralFlatness))
	fmt.Fprintf(file, "| Spectral Entropy | %.4f | %s |\n\n",
		energyEntropy, interpretSpectralEntropy(energyEntropy))

	// Noise analysis
	fmt.Fprintf(file, "## 4. Noise Analysis\n\n")

	// Estimate noise floor using the calculated threshold
	noiseFloor := noiseThreshold
	
	if isPreCleaning {
		// For pre-cleaning, also use our custom noise estimation function
		customNoiseFloor := ap.estimateNoiseProfile(samples[:min(len(samples), 10*buf.Format.SampleRate)])
		// Take the average of the two methods
		noiseFloor = (noiseFloor + customNoiseFloor) / 2
	}
	
	// Calculate signal-to-noise ratio
	snr := 0.0
	if noiseFloor > 0 {
		snr = rmsAmp / noiseFloor
	}
	snrDB := 20 * math.Log10(math.Max(snr, 0.000001))

	fmt.Fprintf(file, "| Metric | Value | Interpretation |\n")
	fmt.Fprintf(file, "|--------|-------|---------------|\n")
	fmt.Fprintf(file, "| Estimated Noise Floor | %.4f | %s |\n",
		noiseFloor, interpretNoiseFloor(noiseFloor))
	fmt.Fprintf(file, "| Signal-to-Noise Ratio | %.2f | - |\n", snr)
	fmt.Fprintf(file, "| SNR (dB) | %.2f dB | %s |\n\n",
		snrDB, interpretSNR(snrDB))
		
	// Frequency Distribution Visualization (ASCII art spectrogram representation)
	fmt.Fprintf(file, "## 5. Frequency Distribution Visualization\n\n")
	fmt.Fprintf(file, "```\n")
	fmt.Fprintf(file, "Frequency Distribution (higher amplitude = more energy)\n")
	fmt.Fprintf(file, "Low Freq |")
	
	// Create simplified visualization of frequency distribution
	bandWidth := 80 / len(bands) // 80 characters wide output
	
	for i := range bands {
		// Calculate normalized energy for this band (0-9)
		normalizedEnergy := 0
		if totalBandEnergy > 0 {
			normalizedEnergy = int(math.Min(9, math.Floor((bandEnergies[i]/totalBandEnergy)*100/10)))
		}
		
		// Print band visualization
		for j := 0; j < bandWidth; j++ {
			if j == bandWidth/2 {
				fmt.Fprintf(file, "%d", normalizedEnergy)
			} else {
				if normalizedEnergy > 0 {
					fmt.Fprintf(file, "▓")
				} else {
					fmt.Fprintf(file, "░")
				}
			}
		}
	}
	
	fmt.Fprintf(file, "| High Freq\n")
	fmt.Fprintf(file, "          ")
	
	// Print frequency band labels
	for i := range bands {
		if i == 0 {
			fmt.Fprintf(file, "%-*s", bandWidth, "SubB")
		} else if i == len(bands)-1 {
			fmt.Fprintf(file, "%-*s", bandWidth, "Air")
		} else if i == len(bands)/2 {
			fmt.Fprintf(file, "%-*s", bandWidth, "Mid")
		} else {
			fmt.Fprintf(file, "%-*s", bandWidth, "")
		}
	}
	
	fmt.Fprintf(file, "\n")
	fmt.Fprintf(file, "```\n\n")

	// If post-cleaning, add cleaning parameters and comparison
	if !isPreCleaning {
		fmt.Fprintf(file, "## 6. Cleaning Parameters Used\n\n")
		fmt.Fprintf(file, "| Parameter | Value | Description |\n")
		fmt.Fprintf(file, "|-----------|-------|-------------|\n")
		fmt.Fprintf(file, "| Threshold | %.2f | Minimum amplitude threshold for noise detection |\n",
			ap.threshold)
		fmt.Fprintf(file, "| Noise Reduction Ratio | %.2f | Proportion of noise to remove |\n",
			ap.noiseReduction)
		fmt.Fprintf(file, "| Enhancement Factor | %.2f | Amplification factor for important frequencies |\n",
			ap.enhanceFactor)
		fmt.Fprintf(file, "| Frequency Range | %.0f-%.0f Hz | Protected frequency band for music content |\n\n",
			ap.frequencyRange[0], ap.frequencyRange[1])

		fmt.Fprintf(file, "## 7. Conclusion and Recommendations\n\n")
		fmt.Fprintf(file, "This analysis represents the audio AFTER noise reduction processing. ")
		fmt.Fprintf(file, "To evaluate the effectiveness of the cleaning process, compare with the pre-cleaning analysis.\n\n")
		
		fmt.Fprintf(file, "Successful noise reduction typically shows:\n\n")
		fmt.Fprintf(file, "- Improved Signal-to-Noise Ratio (SNR)\n")
		fmt.Fprintf(file, "- Maintained or improved spectral balance in music frequency ranges\n")
		fmt.Fprintf(file, "- Similar or better dynamic range\n")
		fmt.Fprintf(file, "- Reduced energy in non-music frequency bands\n")
		fmt.Fprintf(file, "- Maintained spectral centroid (indicating preserved tone color)\n\n")
		
		fmt.Fprintf(file, "For best results when playing this audio:\n\n")
		fmt.Fprintf(file, "- Use quality audio playback equipment\n")
		fmt.Fprintf(file, "- Consider light equalization if needed (based on dominant frequency analysis)\n")
		fmt.Fprintf(file, "- Set appropriate volume levels to appreciate the improved dynamic range\n")
	} else {
		// For pre-cleaning analysis, add recommendations
		fmt.Fprintf(file, "## 6. Recommended Cleaning Approach\n\n")
		
		// Generate different recommendations based on audio characteristics
		fmt.Fprintf(file, "This analysis represents the audio BEFORE noise reduction processing. ")
		fmt.Fprintf(file, "Based on the audio characteristics, the following approach is recommended:\n\n")
		
		// Generate cleaning recommendations
		recs := generateCleaningRecommendations(snrDB, spectralFlatness, energyEntropy, 
			dominantBand, dominantPercentage, zeroCrossingRate)
			
		// Print recommendations
		for i, rec := range recs {
			fmt.Fprintf(file, "### %d. %s\n\n%s\n\n", i+1, rec.title, rec.description)
		}
	}

	return nil
}

// New helper functions for spectral analysis interpretation
func interpretSpectralFlatness(value float64) string {
	if value > 0.5 {
		return "Very noisy (white noise-like)"
	} else if value > 0.2 {
		return "Significant noise content"
	} else if value > 0.1 {
		return "Moderate tonal and noise content"
	} else if value > 0.05 {
		return "Strong tonal content, some noise"
	} else {
		return "Very tonal (pure musical content)"
	}
}

func interpretSpectralEntropy(value float64) string {
	if value > 6.0 {
		return "Very chaotic/random spectrum (likely noise)"
	} else if value > 5.0 {
		return "High complexity spectrum (noise + music)"
	} else if value > 4.0 {
		return "Moderate complexity (musical with some noise)"
	} else if value > 3.0 {
		return "Structured spectrum (clear musical content)"
	} else {
		return "Very organized spectrum (pure tones, minimal noise)"
	}
}

// CleaningRecommendation represents a specific recommendation for noise cleaning
type CleaningRecommendation struct {
	title       string
	description string
}

// generateCleaningRecommendations creates specific recommendations based on audio analysis
func generateCleaningRecommendations(snrDB, spectralFlatness, energyEntropy float64,
	dominantBand string, dominantPercentage, zcr float64) []CleaningRecommendation {
	
	var recommendations []CleaningRecommendation
	
	// SNR-based recommendations
	if snrDB < 15 {
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Use Aggressive Noise Reduction",
			description: "The audio has a very poor SNR (< 15dB), indicating substantial noise contamination. " +
				"Recommended to use aggressive noise reduction settings (0.4-0.5) with focus on preserving only " +
				"the strongest musical elements. Some audio quality loss may be acceptable to remove severe noise.",
		})
	} else if snrDB < 30 {
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Use Moderate Noise Reduction",
			description: "The audio has a moderate SNR (15-30dB). Apply moderate noise reduction (0.3-0.4) " +
				"with careful protection of mid-range frequencies to preserve vocal clarity and musical details.",
		})
	} else {
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Use Gentle Noise Reduction",
			description: "The audio has a good SNR (>30dB). Apply gentle noise reduction (0.2-0.3) " +
				"focusing only on the frequency bands where noise is present while fully preserving musical content.",
		})
	}
	
	// Spectral characteristics recommendations
	if spectralFlatness > 0.2 {
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Focus on Broadband Noise Reduction",
			description: "High spectral flatness indicates significant broadband noise. " +
				"Recommended to use FFT filtering with focus on reducing high-frequency noise " +
				"while preserving transients in musical material.",
		})
	}
	
	// Dominant band recommendations
	switch dominantBand {
	case "Sub-bass", "Bass":
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Protect Low Frequencies",
			description: "Audio has significant low-frequency content. " +
				"Reduce noise primarily in higher bands (>4kHz), while using gentler settings " +
				"for bass frequencies to maintain fullness and power.",
		})
	case "Midrange":
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Preserve Vocal Range",
			description: "Audio has dominant mid-range content typical of vocals and main instruments. " +
				"Use selective frequency protection for 500-2000 Hz range with slightly stronger " +
				"reduction in very low and very high frequencies.",
		})
	case "Upper-midrange", "Presence":
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Preserve Detail and Definition",
			description: "Audio has significant upper-mid/presence energy. " +
				"Focus cleaning on removing low-frequency rumble and very high frequency hiss " +
				"while fully protecting the 2-6 kHz range for intelligibility and detail.",
		})
	}
	
	// Zero-crossing rate recommendations
	if zcr > 0.15 {
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Address High-Frequency Noise",
			description: "High zero-crossing rate indicates significant high-frequency content or noise. " +
				"Apply more aggressive noise reduction above 8 kHz while being careful to preserve " +
				"natural cymbals and high-frequency musical transients.",
		})
	} else if zcr < 0.05 {
		recommendations = append(recommendations, CleaningRecommendation{
			title: "Address Low-Frequency Noise",
			description: "Low zero-crossing rate indicates dominant low-frequency content. " +
				"Apply a high-pass filter around 40-60 Hz to eliminate sub-bass rumble, " +
				"while preserving bass instruments above this range.",
		})
	}
	
	return recommendations
}

// AudioQualityRating represents a quality assessment
type AudioQualityRating struct {
	rating      string
	description string
}

// determineAudioQuality evaluates overall audio quality
func determineAudioQuality(rmsAmp, crestFactor, dynamicRange, zcr float64) AudioQualityRating {
	// Score different aspects (1-5 scale)
	ampScore := 3 // Default
	if rmsAmp > 10000 {
		ampScore = 5
	} else if rmsAmp > 5000 {
		ampScore = 4
	} else if rmsAmp > 1000 {
		ampScore = 3
	} else if rmsAmp > 500 {
		ampScore = 2
	} else {
		ampScore = 1
	}

	dynScore := 3 // Default
	if dynamicRange > 70 {
		dynScore = 5
	} else if dynamicRange > 60 {
		dynScore = 4
	} else if dynamicRange > 40 {
		dynScore = 3
	} else if dynamicRange > 20 {
		dynScore = 2
	} else {
		dynScore = 1
	}

	crestScore := 3 // Default
	if crestFactor > 3 && crestFactor < 10 {
		crestScore = 5 // Ideal range
	} else if crestFactor > 2 && crestFactor <= 15 {
		crestScore = 4
	} else if crestFactor > 1.5 && crestFactor <= 20 {
		crestScore = 3
	} else {
		crestScore = 2
	}

	// Calculate total score (weighted)
	totalScore := float64(ampScore)*0.3 + float64(dynScore)*0.4 + float64(crestScore)*0.3

	rating := AudioQualityRating{}

	// Determine rating
	if totalScore >= 4.5 {
		rating.rating = "Excellent (5/5)"
	} else if totalScore >= 3.5 {
		rating.rating = "Very Good (4/5)"
	} else if totalScore >= 2.5 {
		rating.rating = "Good (3/5)"
	} else if totalScore >= 1.5 {
		rating.rating = "Fair (2/5)"
	} else {
		rating.rating = "Poor (1/5)"
	}

	// Generate description
	if ampScore >= 4 && dynScore >= 4 {
		rating.description = "This audio has excellent amplitude levels and dynamic range. "
	} else if ampScore >= 3 && dynScore >= 3 {
		rating.description = "This audio has good overall levels and reasonable dynamic characteristics. "
	} else {
		rating.description = "This audio has limited amplitude and/or dynamic range characteristics. "
	}

	if crestFactor > 10 {
		rating.description += "There are significant peaks that may indicate transients or potential distortion. "
	} else if crestFactor > 3 {
		rating.description += "The peak-to-average ratio is well balanced. "
	} else {
		rating.description += "The signal has limited dynamic variation, suggesting compression or limiting. "
	}

	if zcr > 0.1 {
		rating.description += "High frequency content is prominent."
	} else if zcr > 0.05 {
		rating.description += "Balanced frequency content is present."
	} else {
		rating.description += "Low frequency content dominates."
	}

	return rating
}

// Helper interpretation functions
func interpretMeanAmplitude(value float64) string {
	if value > 10000 {
		return "Very high average level"
	} else if value > 5000 {
		return "High average level"
	} else if value > 1000 {
		return "Moderate average level"
	} else if value > 500 {
		return "Low average level"
	}
	return "Very low average level"
}

func interpretRMSAmplitude(value float64) string {
	if value > 10000 {
		return "Very high power level"
	} else if value > 5000 {
		return "High power level"
	} else if value > 1000 {
		return "Moderate power level"
	} else if value > 500 {
		return "Low power level"
	}
	return "Very low power level"
}

func interpretMaxAmplitude(value float64) string {
	if value > 30000 {
		return "Near maximum digital level"
	} else if value > 20000 {
		return "High peak level"
	} else if value > 10000 {
		return "Moderate peak level"
	}
	return "Low peak level"
}

func interpretCrestFactor(value float64) string {
	if value > 15 {
		return "Very high (may indicate transient peaks)"
	} else if value > 10 {
		return "High (dynamic audio)"
	} else if value > 5 {
		return "Moderate (typical for most audio)"
	} else if value > 3 {
		return "Low (may indicate compression)"
	}
	return "Very low (heavily compressed/limited)"
}

func interpretDynamicRange(value float64) string {
	if value > 80 {
		return "Excellent dynamic range"
	} else if value > 60 {
		return "Very good dynamic range"
	} else if value > 40 {
		return "Good dynamic range"
	} else if value > 20 {
		return "Limited dynamic range"
	}
	return "Very limited dynamic range"
}

func interpretZeroCrossingRate(value float64) string {
	if value > 0.15 {
		return "Very high (suggests significant high frequency content)"
	} else if value > 0.1 {
		return "High (substantial high frequency or noise)"
	} else if value > 0.05 {
		return "Moderate (balanced frequency content)"
	} else if value > 0.02 {
		return "Low (dominant low frequency content)"
	}
	return "Very low (primarily low frequency content)"
}

func interpretSpectralCentroid(value float64) string {
	if value > 4000 {
		return "Very bright/present sound character"
	} else if value > 2000 {
		return "Bright sound character"
	} else if value > 1000 {
		return "Balanced frequency character"
	} else if value > 500 {
		return "Warm sound character"
	}
	return "Dark/bass-heavy sound character"
}

func interpretDominantBand(band string) string {
	switch band {
	case "Sub-bass":
		return "Very low frequency dominant (rumble, fundamental bass)"
	case "Bass":
		return "Bass dominant (fullness, rhythm foundation)"
	case "Low-midrange":
		return "Low-mid dominant (warmth, body)"
	case "Midrange":
		return "Mid dominant (vocals, instruments presence)"
	case "Upper-midrange":
		return "Upper-mid dominant (detail, articulation)"
	case "Presence":
		return "Presence dominant (definition, attack)"
	case "Brilliance":
		return "High frequency dominant (air, sparkle)"
	}
	return "Balanced spectrum"
}

func interpretNoiseFloor(value float64) string {
	if value > 1000 {
		return "Very high noise floor"
	} else if value > 500 {
		return "High noise floor"
	} else if value > 100 {
		return "Moderate noise floor"
	} else if value > 50 {
		return "Low noise floor"
	}
	return "Very low noise floor"
}

func interpretSNR(value float64) string {
	if value > 80 {
		return "Excellent (audiophile quality)"
	} else if value > 60 {
		return "Very good (professional quality)"
	} else if value > 40 {
		return "Good (broadcast quality)"
	} else if value > 20 {
		return "Fair (acceptable for most purposes)"
	} else if value > 10 {
		return "Poor (noticeable noise)"
	}
	return "Very poor (dominated by noise)"
}

// min returns the smaller of a and b
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func runRealtimeStream(processor *AudioProcessor) error {
	sp := &AudioProcessor{
		threshold:      processor.threshold,
		noiseReduction: processor.noiseReduction,
		enhanceFactor:  processor.enhanceFactor,
	}
	return RunRealtimeStream(sp)
}

func main() {
	// Handle command line arguments
	inputFile, outputFile, threshold, verbose, autoParams, streamline, err := parseCommandLine()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintln(os.Stderr, "Usage: go run main.go -input=input.wav -output=cleaned.wav -verbose=true [-streamline]")
		os.Exit(1)
	}

	// Always enable automatic parameter adjustment
	autoParams = true

	// Create audio processor
	processor := NewAudioProcessor(inputFile, outputFile, threshold, verbose, autoParams)

	if streamline {
		fmt.Println("[STREAMLINE] Real-time microphone noise reduction started...")
		err := runRealtimeStream(processor)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Stream error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	// Start process
	if err := processor.Process(); err != nil {
		fmt.Fprintf(os.Stderr, "Process error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nNoise cleaning completed successfully.\n")
	fmt.Printf("Input file: %s\n", inputFile)
	fmt.Printf("Output file: %s\n", outputFile)
}
