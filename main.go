package main

import (
	"errors"
	"flag"
	"fmt"
	"math"
	"math/cmplx"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/go-audio/audio"
	"github.com/go-audio/wav"
	"gonum.org/v1/gonum/dsp/fourier"
)

// AudioProcessor, ses işleme işlemlerini yöneten ara yapı
type AudioProcessor struct {
	inputFile        string
	outputFile       string
	threshold        float64
	verbose          bool
	enhanceFactor    float64    // Müzik sinyallerini güçlendirme faktörü
	noiseReduction   float64    // Gürültü azaltma oranı (0.0-1.0 arası)
	preserveHarmonic bool       // Harmonik frekansları koruma
	frequencyRange   [2]float64 // Korunacak frekans aralığı (Hz)
}

// NewAudioProcessor, yeni bir AudioProcessor yapısı oluşturur
func NewAudioProcessor(inputFile, outputFile string, threshold float64, verbose bool) *AudioProcessor {
	return &AudioProcessor{
		inputFile:        inputFile,
		outputFile:       outputFile,
		threshold:        threshold,
		verbose:          verbose,
		enhanceFactor:    1.1,                   // Müzik sinyallerini sadece %10 güçlendir (daha az agresif)
		noiseReduction:   0.4,                   // Gürültüyü sadece %40 azalt (daha az agresif)
		preserveHarmonic: true,                  // Harmonik frekansları koru
		frequencyRange:   [2]float64{60, 14000}, // Daha geniş frekans aralığı (60Hz-14kHz)
	}
}

// Process, ses işleme sürecini başlatır
func (ap *AudioProcessor) Process() error {
	startTime := time.Now()

	fmt.Printf("İşlem başlatılıyor...\n")
	fmt.Printf("Giriş dosyası: %s\n", ap.inputFile)
	fmt.Printf("Çıkış dosyası: %s\n", ap.outputFile)
	fmt.Printf("Eşik değeri: %.2f\n", ap.threshold)
	fmt.Printf("Gürültü azaltma oranı: %.2f\n", ap.noiseReduction)
	fmt.Printf("Sinyal güçlendirme: %.2f\n", ap.enhanceFactor)
	fmt.Printf("Korunan frekans aralığı: %.0f Hz - %.0f Hz\n", ap.frequencyRange[0], ap.frequencyRange[1])
	fmt.Println("----------------------------------------")

	// 1. Ses dosyasını oku
	fmt.Printf("[%3d%%] Ses dosyası okunuyor...\n", 0)
	buf, err := ap.readWavFile()
	if err != nil {
		return fmt.Errorf("ses dosyası okuma hatası: %w", err)
	}
	fmt.Printf("[%3d%%] Ses dosyası başarıyla okundu. Süre: %.2f sn, Kanal sayısı: %d, Örnekleme hızı: %d Hz\n",
		25,
		time.Since(startTime).Seconds(),
		buf.Format.NumChannels,
		buf.Format.SampleRate)

	// 2. Gürültü temizleme işlemini gerçekleştir
	fmt.Printf("[%3d%%] Gürültü temizleme işlemi başlatılıyor...\n", 25)
	cleanedBuffer, err := ap.applyNoiseCancellation(buf)
	if err != nil {
		return fmt.Errorf("gürültü temizleme işlemi hatası: %w", err)
	}
	fmt.Printf("[%3d%%] Gürültü temizleme işlemi tamamlandı. Süre: %.2f sn\n",
		75,
		time.Since(startTime).Seconds())

	// 3. Temizlenmiş sesi kaydet
	fmt.Printf("[%3d%%] Temizlenmiş ses kaydediliyor...\n", 75)
	if err := ap.saveWavFile(cleanedBuffer); err != nil {
		return fmt.Errorf("ses dosyası kaydetme hatası: %w", err)
	}
	fmt.Printf("[%3d%%] Temizlenmiş ses başarıyla kaydedildi. Toplam işlem süresi: %.2f sn\n",
		100,
		time.Since(startTime).Seconds())
	fmt.Println("----------------------------------------")

	return nil
}

// readWavFile, bir WAV ses dosyasını okur ve PCM buffer döndürür
func (ap *AudioProcessor) readWavFile() (*audio.IntBuffer, error) {
	file, err := os.Open(ap.inputFile)
	if err != nil {
		return nil, fmt.Errorf("dosya açılamadı: %w", err)
	}
	defer file.Close()

	decoder := wav.NewDecoder(file)
	if !decoder.IsValidFile() {
		return nil, errors.New("geçersiz WAV dosyası")
	}

	buf, err := decoder.FullPCMBuffer()
	if err != nil {
		return nil, fmt.Errorf("PCM buffer okunamadı: %w", err)
	}

	return buf, nil
}

// applyNoiseCancellation, FFT kullanarak gürültü temizleme işlemini uygular
func (ap *AudioProcessor) applyNoiseCancellation(buf *audio.IntBuffer) (*audio.IntBuffer, error) {
	totalSamples := len(buf.Data)
	progressStep := totalSamples / 10 // Her %10'luk ilerleme için bildirim

	fmt.Printf("Toplam işlenecek örnek sayısı: %d\n", totalSamples)

	// PCM verilerini float64 dizisine dönüştür
	samples := make([]float64, totalSamples)
	for i, sample := range buf.Data {
		samples[i] = float64(sample)

		// İlerleme durumunu göster
		if ap.verbose && i > 0 && i%progressStep == 0 {
			progress := int((float64(i) / float64(totalSamples)) * 100)
			fmt.Printf("\r[%3d%%] Veriler dönüştürülüyor... %d/%d",
				25+(progress/5),
				i,
				totalSamples)
		}
	}
	if ap.verbose {
		fmt.Println()
	}

	// Sesi segmentlere bölüp her birine FFT uygula (örtüşmeli pencereler)
	windowSize := 4096            // Daha büyük pencere boyutu (daha iyi frekans çözünürlüğü için)
	overlap := windowSize * 3 / 4 // Daha fazla örtüşme (%75)
	hopSize := windowSize - overlap

	// Uygun bir pencere boyutu için düzenleme
	numWindows := 1 + (totalSamples-windowSize)/hopSize

	fmt.Printf("Sinyal %d segmente bölünerek işlenecek (pencere boyutu: %d, örtüşme: %d)\n",
		numWindows, windowSize, overlap)

	// Sonuç vektörü
	result := make([]float64, totalSamples)

	// Gürültü profilini belirlemek için ilk pencereyi kullan
	noiseProfile := ap.estimateNoiseProfile(samples[:windowSize])
	fmt.Printf("Gürültü profili oluşturuldu. Ortalama gürültü seviyesi: %.2f\n", noiseProfile)

	// Her pencereyi ayrı ayrı işle
	for i := 0; i < numWindows; i++ {
		startIdx := i * hopSize
		endIdx := startIdx + windowSize

		if endIdx > totalSamples {
			endIdx = totalSamples
		}

		segment := samples[startIdx:endIdx]

		// Pencere fonksiyonu uygula (Hanning window)
		segmentWindowed := ap.applyWindow(segment)

		// FFT ile temizleme işlemi
		cleanSegment := ap.denoiseWithFFT(segmentWindowed, noiseProfile, float64(buf.Format.SampleRate))

		// Temizlenmiş segmenti sonuç vektörüne ekle (örtüşme bölgelerini ortalama ile birleştir)
		for j := 0; j < len(cleanSegment); j++ {
			if startIdx+j < totalSamples {
				if i > 0 && startIdx+j < startIdx+overlap { // Örtüşme bölgesindeyse
					weight := float64(j) / float64(overlap) // Örtüşme ağırlığı
					result[startIdx+j] = result[startIdx+j]*(1-weight) + cleanSegment[j]*weight
				} else {
					result[startIdx+j] = cleanSegment[j]
				}
			}
		}

		// İlerleme bildirimi
		if ap.verbose && i%10 == 0 {
			progress := int((float64(i) / float64(numWindows)) * 100)
			fmt.Printf("\r[%3d%%] Segment işleniyor %d/%d",
				30+(progress/2),
				i,
				numWindows)
		}
	}

	if ap.verbose {
		fmt.Println()
	}

	// Temizlenmiş ses verilerini int formatına geri dönüştür
	fmt.Printf("Temizlenmiş veri dönüşümü yapılıyor...\n")

	// Normalize et (clipping önlemek için)
	maxValue := 0.0
	for _, v := range result {
		if math.Abs(v) > maxValue {
			maxValue = math.Abs(v)
		}
	}

	// Eğer maksimum değer çok büyükse, normalize et
	scaleFactor := 1.0
	if maxValue > float64(math.MaxInt16) { // Int32 yerine daha makul bir sınır (16-bit PCM)
		scaleFactor = float64(math.MaxInt16) / maxValue * 0.95 // %5 marj bırak
	}

	for i := range result {
		if i < len(buf.Data) {
			buf.Data[i] = int(result[i] * scaleFactor)
		}

		// İlerleme durumunu göster
		if ap.verbose && i > 0 && i%progressStep == 0 {
			progress := int((float64(i) / float64(totalSamples)) * 100)
			fmt.Printf("\r[%3d%%] Veriler geri dönüştürülüyor... %d/%d",
				75+(progress/4),
				i,
				totalSamples)
		}
	}
	if ap.verbose {
		fmt.Println()
	}

	return buf, nil
}

// estimateNoiseProfile, sinyal içindeki gürültü seviyesini tahmin eder
func (ap *AudioProcessor) estimateNoiseProfile(samples []float64) float64 {
	// Örnekleri frekans domaininde inceleyerek gürültü seviyesini tahmin et
	fft := fourier.NewFFT(len(samples))
	freqDomain := fft.Coefficients(nil, samples)

	// Gürültü seviyesini tahmin etmek için daha dengeli bir yaklaşım
	var noiseSum float64
	var count int

	// Yüksek frekansları (üst %20) ve düşük frekansları (alt %5) inceleyerek gürültü tahmini yap
	// Bu yaklaşım daha iyi bir gürültü profili çıkarmaya yardımcı olur

	// Yüksek frekanslar (tipik arka plan gürültüsü)
	for i := len(freqDomain) * 8 / 10; i < len(freqDomain); i++ {
		noiseSum += cmplx.Abs(freqDomain[i])
		count++
	}

	// Çok düşük frekanslar (genellikle ortam gürültüsü)
	for i := 1; i < len(freqDomain)*5/100; i++ {
		noiseSum += cmplx.Abs(freqDomain[i])
		count++
	}

	// Ortalama gürültü seviyesi
	avgNoiseLevel := noiseSum / float64(count)

	// Daha düşük bir eşik değeri kullan - sokak müziği kayıtları için
	scaledNoise := avgNoiseLevel * 0.3 // Sadece %30 gürültü tahmini (daha az agresif)

	return math.Max(scaledNoise, ap.threshold*0.2) // Eşik değerinin %20'si veya gürültü tahmini
}

// applyWindow, zaman domainindeki sinyali pencere fonksiyonu ile çarpar
func (ap *AudioProcessor) applyWindow(samples []float64) []float64 {
	result := make([]float64, len(samples))

	// Hanning penceresi uygula (pencere uçlarında yumuşak geçiş için)
	for i := 0; i < len(samples); i++ {
		// w(n) = 0.5 * (1 - cos(2π*n/(N-1)))
		windowCoeff := 0.5 * (1.0 - math.Cos(2.0*math.Pi*float64(i)/float64(len(samples)-1)))
		result[i] = samples[i] * windowCoeff
	}

	return result
}

// denoiseWithFFT, FFT uygulayarak frekans domaininde gürültüyü filtreler
func (ap *AudioProcessor) denoiseWithFFT(samples []float64, noiseProfile float64, sampleRate float64) []float64 {
	// Örneklem sayısının 2'nin kuvveti olmasını sağla (FFT için)
	paddedLength := nextPowerOfTwo(len(samples))
	paddedSamples := make([]float64, paddedLength)
	copy(paddedSamples, samples)

	if ap.verbose {
		fmt.Printf("Segment uzunluğu: %d, FFT için optimize edilmiş uzunluk: %d\n", len(samples), paddedLength)
	}

	// FFT hesapla
	fft := fourier.NewFFT(paddedLength)
	freqDomain := fft.Coefficients(nil, paddedSamples)

	// Spektral filtreleme uygula
	var enhancedCount, filteredCount int

	// Hz cinsinden frekans aralığını hesapla
	freqResolution := sampleRate / float64(paddedLength)
	minFreqIndex := int(ap.frequencyRange[0] / freqResolution)
	maxFreqIndex := int(ap.frequencyRange[1] / freqResolution)
	if maxFreqIndex > len(freqDomain)/2 {
		maxFreqIndex = len(freqDomain) / 2
	}

	// Düşük gürültü eşiği ve yüksek gürültü eşiği belirle
	lowNoiseThreshold := noiseProfile * 1.5
	highNoiseThreshold := noiseProfile * 4.0

	// Faz hassasiyeti için original faz bilgisini koru
	phases := make([]float64, len(freqDomain))
	for i := range freqDomain {
		phases[i] = cmplx.Phase(freqDomain[i])
	}

	for i := range freqDomain {
		magnitude := cmplx.Abs(freqDomain[i])
		phase := phases[i]

		// Eğer frekans müzik için önemli aralıktaysa (60Hz-14kHz) ve yeterince güçlüyse koru
		inMusicRange := (i >= minFreqIndex && i <= maxFreqIndex) ||
			(i >= paddedLength-maxFreqIndex && i <= paddedLength-minFreqIndex)

		var newMagnitude float64

		if inMusicRange && magnitude > highNoiseThreshold {
			// Önemli müzik frekanslarını hafifçe güçlendir
			newMagnitude = magnitude * ap.enhanceFactor
			enhancedCount++
		} else if magnitude < lowNoiseThreshold {
			// Sadece çok düşük sinyalleri filtrele ve çok agresif olma
			// Tamamen sıfırlamak yerine sadece azalt
			newMagnitude = magnitude * 0.3 // Düşük sinyalleri %70 azalt
			filteredCount++
		} else {
			// Diğer frekansları koru, ancak çok düşük veya çok yüksek değilse
			// biraz gürültü azaltma uygula
			if magnitude < highNoiseThreshold {
				// Yumuşak bir geçiş eğrisi uygula
				ratio := (magnitude - lowNoiseThreshold) / (highNoiseThreshold - lowNoiseThreshold)
				reduction := ap.noiseReduction * (1.0 - ratio) // Azaltma oranını hesapla
				newMagnitude = magnitude * (1.0 - reduction)
				filteredCount++
			} else {
				// Yüksek güçlü sinyalleri koru
				newMagnitude = magnitude
			}
		}

		// Frekans bileşenini güncelle (güç değişti, faz aynı kaldı)
		freqDomain[i] = cmplx.Rect(newMagnitude, phase)
	}

	if ap.verbose {
		filteredPercentage := (float64(filteredCount) / float64(len(freqDomain))) * 100
		enhancedPercentage := (float64(enhancedCount) / float64(len(freqDomain))) * 100
		fmt.Printf("Filtreleme uygulandı. Filtrelenen: %d (%%%.2f), Güçlendirilen: %d (%%%.2f)\n",
			filteredCount, filteredPercentage, enhancedCount, enhancedPercentage)
	}

	// Ters FFT ile temizlenmiş sinyali elde et
	cleanSignal := fft.Sequence(nil, freqDomain)

	// Orijinal boyuta geri dön
	return cleanSignal[:len(samples)]
}

// calculateReductionFactor, gürültü azaltma faktörünü hesaplar
func (ap *AudioProcessor) calculateReductionFactor(magnitude, noiseProfile float64) float64 {
	// Ne kadar gürültü olduğunu tahmin et (1.0 = tamamen gürültü, 0.0 = tamamen sinyal)
	// Müzik sinyalleri için, gürültü olma olasılığını azalt

	if magnitude <= noiseProfile {
		return ap.noiseReduction // Kesin gürültü - yüksek azaltma
	}

	// Gürültü olma olasılığını hesapla (noiseProfile'a yakınlık bazında)
	noiseProbability := noiseProfile / magnitude
	if noiseProbability > 1.0 {
		noiseProbability = 1.0
	}

	// Yumuşak azaltma uygula
	return noiseProbability * ap.noiseReduction
}

// saveWavFile, temizlenmiş PCM verilerini WAV dosyası olarak kaydeder
func (ap *AudioProcessor) saveWavFile(buf *audio.IntBuffer) error {
	outFile, err := os.Create(ap.outputFile)
	if err != nil {
		return fmt.Errorf("çıktı dosyası oluşturulamadı: %w", err)
	}
	defer outFile.Close()

	// Encoder oluştur, Format yapısından gerekli bilgileri al
	encoder := wav.NewEncoder(
		outFile,
		buf.Format.SampleRate,
		int(buf.SourceBitDepth),
		buf.Format.NumChannels,
		1, // PCM formatı için 1 (WAV audio formatı)
	)
	defer encoder.Close()

	if err := encoder.Write(buf); err != nil {
		return fmt.Errorf("veri yazma hatası: %w", err)
	}

	return nil
}

// nextPowerOfTwo, verilen sayıdan büyük veya eşit en küçük 2^n değerini döndürür
func nextPowerOfTwo(n int) int {
	p := 1
	for p < n {
		p *= 2
	}
	return p
}

// parseCommandLine, komut satırı argümanlarını işler
func parseCommandLine() (string, string, float64, bool, error) {
	var inputFile string
	var outputFile string
	var threshold float64
	var verbose bool
	var enhanceFactor float64
	var noiseReduction float64
	var minFreq, maxFreq float64

	flag.StringVar(&inputFile, "input", "", "Giriş ses dosyası (WAV formatında)")
	flag.StringVar(&outputFile, "output", "", "Çıkış ses dosyası (temizlenmiş)")
	flag.Float64Var(&threshold, "threshold", 200.0, "Gürültü filtreleme eşik değeri (varsayılan: 200.0)")
	flag.BoolVar(&verbose, "verbose", false, "Detaylı işlem bilgilerini göster")
	flag.Float64Var(&enhanceFactor, "enhance", 1.1, "Müzik sinyallerini güçlendirme faktörü (1.0-2.0 arası)")
	flag.Float64Var(&noiseReduction, "reduction", 0.4, "Gürültü azaltma oranı (0.0-1.0 arası)")
	flag.Float64Var(&minFreq, "min-freq", 60.0, "Korunacak minimum frekans (Hz)")
	flag.Float64Var(&maxFreq, "max-freq", 14000.0, "Korunacak maksimum frekans (Hz)")
	flag.Parse()

	if inputFile == "" {
		return "", "", 0, false, errors.New("giriş dosyası belirtilmedi")
	}

	if outputFile == "" {
		// Giriş dosyasının adını kullanarak otomatik çıkış dosyası adı oluştur
		ext := filepath.Ext(inputFile)
		base := inputFile[:len(inputFile)-len(ext)]
		outputFile = base + "_cleaned" + ext
	}

	// Parametrelerin geçerli aralıklarda olduğunu kontrol et
	if enhanceFactor < 1.0 || enhanceFactor > 2.0 {
		fmt.Fprintf(os.Stderr, "Uyarı: Güçlendirme faktörü (%.2f) önerilen aralık dışında. 1.0-2.0 arası önerilir.\n", enhanceFactor)
	}

	if noiseReduction < 0.0 || noiseReduction > 1.0 {
		fmt.Fprintf(os.Stderr, "Uyarı: Gürültü azaltma oranı (%.2f) geçersiz. 0.0-1.0 arasında bir değer kullanılacak.\n", noiseReduction)
		if noiseReduction < 0.0 {
			noiseReduction = 0.0
		} else if noiseReduction > 1.0 {
			noiseReduction = 1.0
		}
	}

	if minFreq < 20.0 || minFreq > 1000.0 {
		fmt.Fprintf(os.Stderr, "Uyarı: Minimum frekans (%.2f Hz) önerilen aralık dışında. 20-1000 Hz arası önerilir.\n", minFreq)
	}

	if maxFreq < 5000.0 || maxFreq > 20000.0 {
		fmt.Fprintf(os.Stderr, "Uyarı: Maksimum frekans (%.2f Hz) önerilen aralık dışında. 5000-20000 Hz arası önerilir.\n", maxFreq)
	}

	return inputFile, outputFile, threshold, verbose, nil
}

func main() {
	// Komut satırı argümanlarını işle
	inputFile, outputFile, threshold, verbose, err := parseCommandLine()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Hata: %v\n", err)
		fmt.Fprintln(os.Stderr, "Kullanım: go run main.go -input=input.wav -output=cleaned.wav -threshold=200.0 -verbose=true -enhance=1.1 -reduction=0.4 -min-freq=60 -max-freq=14000")
		os.Exit(1)
	}

	// Ses işleyici oluştur
	processor := NewAudioProcessor(inputFile, outputFile, threshold, verbose)

	// Ayarları komut satırından alınan değerlerle güncelle (eğer belirtilmişse)
	flag.Visit(func(f *flag.Flag) {
		switch f.Name {
		case "enhance":
			processor.enhanceFactor, _ = strconv.ParseFloat(f.Value.String(), 64)
		case "reduction":
			processor.noiseReduction, _ = strconv.ParseFloat(f.Value.String(), 64)
		case "min-freq":
			minFreq, _ := strconv.ParseFloat(f.Value.String(), 64)
			processor.frequencyRange[0] = minFreq
		case "max-freq":
			maxFreq, _ := strconv.ParseFloat(f.Value.String(), 64)
			processor.frequencyRange[1] = maxFreq
		}
	})

	// İşlemi başlat
	if err := processor.Process(); err != nil {
		fmt.Fprintf(os.Stderr, "İşlem hatası: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\nGürültü temizleme başarıyla tamamlandı.\n")
	fmt.Printf("Giriş dosyası: %s\n", inputFile)
	fmt.Printf("Çıkış dosyası: %s\n", outputFile)
}
