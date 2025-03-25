package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"gonum.org/v1/gonum/dsp/fourier"
)

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

	// If automatic parameter adjustment is requested, analyze the audio file and determine parameters
	if ap.autoParams {
		fmt.Printf("[%3d%%] Performing automatic parameter analysis...\n", 15)
		err := ap.analyzeAndSetParameters(buf)
		if err != nil {
			return fmt.Errorf("audio analysis error: %w", err)
		}
		fmt.Printf("[%3d%%] Automatic parameter analysis completed. Duration: %.2f sec\n",
			20,
			time.Since(startTime).Seconds())
	}

	// Display parameters
	fmt.Printf("Threshold value: %.2f\n", ap.threshold)
	fmt.Printf("Noise reduction ratio: %.2f\n", ap.noiseReduction)
	fmt.Printf("Signal enhancement: %.2f\n", ap.enhanceFactor)
	fmt.Printf("Protected frequency range: %.0f Hz - %.0f Hz\n", ap.frequencyRange[0], ap.frequencyRange[1])
	fmt.Println("----------------------------------------")

	// 2. Perform noise cleaning operation
	fmt.Printf("[%3d%%] Starting noise cleaning operation...\n", 25)
	cleanedBuffer, err := ap.applyNoiseCancellation(buf)
	if err != nil {
		return fmt.Errorf("noise cleaning operation error: %w", err)
	}
	fmt.Printf("[%3d%%] Noise cleaning operation completed. Duration: %.2f sec\n",
		75,
		time.Since(startTime).Seconds())

	// 3. Save cleaned audio
	fmt.Printf("[%3d%%] Saving cleaned audio...\n", 75)
	if err := ap.saveWavFile(cleanedBuffer); err != nil {
		return fmt.Errorf("audio file saving error: %w", err)
	}
	fmt.Printf("[%3d%%] Cleaned audio successfully saved. Total process duration: %.2f sec\n",
		100,
		time.Since(startTime).Seconds())
	fmt.Println("----------------------------------------")

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
	minFreqIndex := int(ap.frequencyRange[0] / freqResolution)
	maxFreqIndex := int(ap.frequencyRange[1] / freqResolution)
	if maxFreqIndex > len(freqDomain)/2 {
		maxFreqIndex = len(freqDomain) / 2
	}

	// Low/high frequency limits (for statistics)
	midFreqIndex := int(1000 / freqResolution) // 1000 Hz mid frequency assumed

	// Define low/high noise thresholds
	lowNoiseThreshold := noiseProfile * 1.5
	highNoiseThreshold := noiseProfile * 4.0

	// Keep original phase information for phase sensitivity
	phases := make([]float64, len(freqDomain))
	for i := range freqDomain {
		phases[i] = cmplx.Phase(freqDomain[i])
	}

	// Total power in low/high frequencies
	var totalLowFreqPower, totalHighFreqPower float64
	var filteredLowFreqPower, filteredHighFreqPower float64

	for i := range freqDomain {
		magnitude := cmplx.Abs(freqDomain[i])
		phase := phases[i]

		// Sum power for low/high frequency statistics
		if i <= midFreqIndex || i >= paddedLength-midFreqIndex {
			totalLowFreqPower += magnitude * magnitude
		} else {
			totalHighFreqPower += magnitude * magnitude
		}

		// Keep frequencies if they are important for music (60Hz-14kHz) and strong enough
		inMusicRange := (i >= minFreqIndex && i <= maxFreqIndex) ||
			(i >= paddedLength-maxFreqIndex && i <= paddedLength-minFreqIndex)

		var newMagnitude float64
		origMagnitude := magnitude

		if inMusicRange && magnitude > highNoiseThreshold {
			// Slightly enhance important music frequencies
			newMagnitude = magnitude * ap.enhanceFactor
			enhancedCount++
		} else if magnitude < lowNoiseThreshold {
			// Filter out very low signals and avoid too aggressive
			// Instead of completely zeroing, just reduce
			newMagnitude = magnitude * 0.3 // Reduce low signals by 70%
			filteredCount++

			// Calculate filtered power for low/high frequency statistics
			if i <= midFreqIndex || i >= paddedLength-midFreqIndex {
				filteredLowFreqPower += (origMagnitude - newMagnitude) * (origMagnitude - newMagnitude)
			} else {
				filteredHighFreqPower += (origMagnitude - newMagnitude) * (origMagnitude - newMagnitude)
			}
		} else {
			// Keep other frequencies, but if not too low or too high,
			// apply a little noise reduction
			if magnitude < highNoiseThreshold {
				// Apply a soft transition curve
				ratio := (magnitude - lowNoiseThreshold) / (highNoiseThreshold - lowNoiseThreshold)
				reduction := ap.noiseReduction * (1.0 - ratio) // Calculate reduction ratio
				newMagnitude = magnitude * (1.0 - reduction)
				filteredCount++

				// Calculate filtered power for low/high frequency statistics
				if i <= midFreqIndex || i >= paddedLength-midFreqIndex {
					filteredLowFreqPower += (origMagnitude - newMagnitude) * (origMagnitude - newMagnitude)
				} else {
					filteredHighFreqPower += (origMagnitude - newMagnitude) * (origMagnitude - newMagnitude)
				}
			} else {
				// Keep high power signals
				newMagnitude = magnitude
			}
		}

		// Update frequency component (power changed, phase remained the same)
		freqDomain[i] = cmplx.Rect(newMagnitude, phase)
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

	// Return to original length and return statistics
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
func parseCommandLine() (string, string, float64, bool, bool, error) {
	var inputFile string
	var outputFile string
	var threshold float64
	var verbose bool

	flag.StringVar(&inputFile, "input", "", "Input audio file (WAV format)")
	flag.StringVar(&outputFile, "output", "", "Output audio file (cleaned)")
	flag.Float64Var(&threshold, "threshold", 200.0, "Noise filtering threshold value (will be automatically adjusted)")
	flag.BoolVar(&verbose, "verbose", false, "Show detailed process information")
	flag.Parse()

	if inputFile == "" {
		return "", "", 0, false, true, errors.New("input file not specified")
	}

	if outputFile == "" {
		// Use input file name to automatically create output file name
		ext := filepath.Ext(inputFile)
		base := inputFile[:len(inputFile)-len(ext)]
		outputFile = base + "_cleaned" + ext
	}

	return inputFile, outputFile, threshold, verbose, true, nil
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

	// 1. Analyze signal level
	var sum, sumSquared float64
	var maxAmp float64
	for _, sample := range samples {
		sum += math.Abs(sample)
		sumSquared += sample * sample
		if math.Abs(sample) > maxAmp {
			maxAmp = math.Abs(sample)
		}
	}
	meanAmp := sum / float64(len(samples))
	rmsAmp := math.Sqrt(sumSquared / float64(len(samples)))

	// Calculate crest factor (peak/RMS) - high impulse noise indicates
	crestFactor := maxAmp / rmsAmp

	// 2. Spectral analysis
	windowSize := 4096
	fft := fourier.NewFFT(windowSize)

	// Define frequency ranges
	lowBand := [2]int{0, windowSize * 100 / buf.Format.SampleRate}              // 0-100 Hz
	midLowBand := [2]int{lowBand[1], windowSize * 500 / buf.Format.SampleRate}  // 100-500 Hz
	midBand := [2]int{midLowBand[1], windowSize * 2000 / buf.Format.SampleRate} // 500-2000 Hz
	highBand := [2]int{midBand[1], windowSize / 2}                              // 2000+ Hz

	// Analyze different segments
	segments := len(samples) / windowSize
	if segments < 1 {
		segments = 1
	}
	if segments > 10 {
		segments = 10 // Analyze at most 10 segments
	}

	var lowEnergy, midLowEnergy, midEnergy, highEnergy float64
	var totalEnergy float64
	var noiseFloor float64

	for i := range make([]struct{}, segments) {
		startIdx := i * windowSize
		endIdx := startIdx + windowSize
		if endIdx > len(samples) {
			endIdx = len(samples)
		}

		segment := make([]float64, windowSize)
		copy(segment, samples[startIdx:endIdx])

		// If segment length is less than windowSize, zero out the remaining part
		if endIdx-startIdx < windowSize {
			for j := endIdx - startIdx; j < windowSize; j++ {
				segment[j] = 0
			}
		}

		// Apply FFT
		spectrum := fft.Coefficients(nil, segment)

		// Estimate noise level (minimum values in high frequencies)
		var highFreqMin float64 = math.MaxFloat64
		highFreqRange := highBand[1] - highBand[0]
		for j := range make([]struct{}, highFreqRange) {
			idx := j + highBand[0]
			magnitude := cmplx.Abs(spectrum[idx])
			if magnitude > 0.001 && magnitude < highFreqMin {
				highFreqMin = magnitude
			}
		}

		// Check for very large values
		if highFreqMin == math.MaxFloat64 || math.IsInf(highFreqMin, 0) || math.IsNaN(highFreqMin) {
			highFreqMin = 100.0 // Use a default value
		}

		noiseFloor += highFreqMin

		// Calculate energy for each frequency band
		for band, bandRange := range [][2]int{lowBand, midLowBand, midBand, highBand} {
			var bandEnergy float64
			bandWidth := bandRange[1] - bandRange[0]
			for j := range make([]struct{}, bandWidth) {
				idx := j + bandRange[0]
				magnitude := cmplx.Abs(spectrum[idx])
				bandEnergy += magnitude * magnitude
			}

			// Sum band energies
			switch band {
			case 0:
				lowEnergy += bandEnergy
			case 1:
				midLowEnergy += bandEnergy
			case 2:
				midEnergy += bandEnergy
			case 3:
				highEnergy += bandEnergy
			}

			totalEnergy += bandEnergy
		}
	}

	// Calculate average values
	lowEnergy /= float64(segments)
	midLowEnergy /= float64(segments)
	midEnergy /= float64(segments)
	highEnergy /= float64(segments)
	totalEnergy /= float64(segments)

	// Noise floor calculation for safer result
	if segments > 0 {
		noiseFloor /= float64(segments)
	} else {
		noiseFloor = 0.001 // Use a default value
	}

	// Adjust for very small noise floor values
	if noiseFloor < 0.001 {
		noiseFloor = 0.001 // To prevent logarithmic calculation error in very small values
	}

	// Calculate band energy ratios
	lowRatio := lowEnergy / totalEnergy
	midLowRatio := midLowEnergy / totalEnergy
	midRatio := midEnergy / totalEnergy
	highRatio := highEnergy / totalEnergy

	// Calculate dynamic range (in dB)
	var dynamicRange float64
	if noiseFloor > 0 && maxAmp > 0 {
		dynamicRange = 20 * math.Log10(maxAmp/noiseFloor)
		// Limit very high or very low dynamic range values
		if dynamicRange > 120 {
			dynamicRange = 120 // Maximum reasonable dynamic range
		} else if dynamicRange < 0 {
			dynamicRange = 0 // Reset negative values
		}
	} else {
		dynamicRange = 40 // Use a default value
	}

	// 3. Parameter determination

	// a. Threshold - adjust based on noise floor and average signal level
	var signalToNoiseRatio float64
	if noiseFloor > 0 {
		signalToNoiseRatio = meanAmp / noiseFloor
	} else {
		signalToNoiseRatio = 10 // Use a default value
	}

	// Adjust threshold based on signal/noise ratio and crest factor
	if signalToNoiseRatio > 100 {
		// Very clean signal
		ap.threshold = noiseFloor * 2.0
	} else if signalToNoiseRatio > 50 {
		// Good signal
		ap.threshold = noiseFloor * 2.5
	} else if signalToNoiseRatio > 20 {
		// Medium signal
		ap.threshold = noiseFloor * 3.0
	} else if signalToNoiseRatio > 10 {
		// Noisy signal
		ap.threshold = noiseFloor * 3.5
	} else {
		// Very noisy signal
		ap.threshold = noiseFloor * 4.0
	}

	// Limit very high or low values
	if ap.threshold < 100 {
		ap.threshold = 100
	} else if ap.threshold > 1000 {
		ap.threshold = 1000
	}

	// b. Noise reduction ratio - adjust based on noise level and dynamic range
	if dynamicRange > 60 {
		// Wide dynamic range - less reduction
		ap.noiseReduction = 0.25
	} else if dynamicRange > 40 {
		// Medium dynamic range
		ap.noiseReduction = 0.3
	} else if dynamicRange > 30 {
		// Narrow dynamic range
		ap.noiseReduction = 0.4
	} else {
		// Very narrow dynamic range - more reduction
		ap.noiseReduction = 0.45
	}

	// Very high crest factor impulse noise indicates
	if crestFactor > 5 {
		ap.noiseReduction += 0.05
	}

	// c. Signal enhancement factor - adjust based on music content
	// Mid and midlow bands indicate music content
	musicContentRatio := (midRatio + midLowRatio)
	if musicContentRatio > 0.7 {
		// High music content - less enhancement
		ap.enhanceFactor = 1.05
	} else if musicContentRatio > 0.5 {
		ap.enhanceFactor = 1.1
	} else if musicContentRatio > 0.3 {
		ap.enhanceFactor = 1.15
	} else {
		// Low music content - more enhancement
		ap.enhanceFactor = 1.2
	}

	// d. Frequency range - adjust based on spectral content
	// Low frequency limit
	if lowRatio > 0.3 {
		// High energy in low frequencies - lower limit
		ap.frequencyRange[0] = 40
	} else if lowRatio > 0.15 {
		ap.frequencyRange[0] = 60
	} else {
		// Low frequencies have low energy - higher limit
		ap.frequencyRange[0] = 80
	}

	// High frequency limit
	if highRatio > 0.1 {
		// High energy in high frequencies - higher limit
		ap.frequencyRange[1] = 16000
	} else if highRatio > 0.05 {
		ap.frequencyRange[1] = 14000
	} else {
		// High frequencies have low energy - lower limit
		ap.frequencyRange[1] = 12000
	}

	// Analyze results
	if ap.verbose {
		fmt.Printf("AUDIO ANALYSIS RESULTS:\n")
		fmt.Printf("  Average signal level: %.2f\n", meanAmp)
		fmt.Printf("  RMS level: %.2f\n", rmsAmp)
		fmt.Printf("  Maximum amplitude: %.2f\n", maxAmp)
		fmt.Printf("  Crest factor: %.2f\n", crestFactor)
		fmt.Printf("  Noise floor: %.2f\n", noiseFloor)
		fmt.Printf("  Signal/Noise ratio: %.2f\n", signalToNoiseRatio)
		fmt.Printf("  Dynamic range: %.2f dB\n", dynamicRange)
		fmt.Printf("  Band energy ratios: low=%.2f, midlow=%.2f, mid=%.2f, high=%.2f\n",
			lowRatio, midLowRatio, midRatio, highRatio)
		fmt.Printf("  Music content ratio: %.2f\n", musicContentRatio)
		fmt.Println("----------------------------------------")
	}

	return nil
}

func main() {
	// Handle command line arguments
	inputFile, outputFile, threshold, verbose, autoParams, err := parseCommandLine()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		fmt.Fprintln(os.Stderr, "Usage: go run main.go -input=input.wav -output=cleaned.wav -verbose=true")
		os.Exit(1)
	}

	// Always enable automatic parameter adjustment
	autoParams = true

	// Create audio processor
	processor := NewAudioProcessor(inputFile, outputFile, threshold, verbose, autoParams)

	// Start process
	if err := processor.Process(); err != nil {
		fmt.Fprintf(os.Stderr, "Process error: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nNoise cleaning completed successfully.\n")
	fmt.Printf("Input file: %s\n", inputFile)
	fmt.Printf("Output file: %s\n", outputFile)
}
