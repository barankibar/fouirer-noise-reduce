# Fourier Gürültü Azaltma Uygulaması

## Genel Bakış

Bu uygulama iki modda ses gürültü azaltma özellikleri sunar:

1. Dosya tabanlı işleme: Ses dosyalarından gürültü temizleme
2. Gerçek zamanlı işleme: Mikrofon girişinden gürültüyü gerçek zamanlı azaltma

## Özellikler

### Dosya İşleme Modu

- WAV ses dosyalarını okur
- Detaylı ses analizi yapar
- Fourier dönüşümü kullanarak gürültü azaltma uygular
- Kapsamlı analiz raporları oluşturur
- Temizlenmiş sesi yeni bir dosyaya kaydeder

### Gerçek Zamanlı İşleme Modu

- Mikrofon sesini yakalar
- Gerçek zamanlı gürültü azaltma uygular
- Temizlenmiş sesi hoparlörlere çıkarır
- Gürültü azaltma için hareketli ortalama filtresi kullanır

## Teknik Detaylar

### Ses İşleme Parametreleri

- Örnekleme Hızı: 44.1 kHz
- Kanallar: Mono (1 kanal)
- Kare Boyutu: 1024 örnek
- İşleme Penceresi: 5 örnek (gerçek zamanlı mod için)

### Gürültü Azaltma Algoritması

1. Dosya Modu:
   - Frekans analizi için FFT (Hızlı Fourier Dönüşümü) kullanır
   - Uyarlanabilir eşikleme uygular
   - Harmonik frekansları korur
   - Müzik sinyallerini güçlendirir

2. Gerçek Zamanlı Mod:
   - Hareketli ortalama filtresi kullanır
   - Eşik tabanlı gürültü azaltma uygular
   - Eşik üzerindeki sinyalleri güçlendirir

## Kullanım

### Komut Satırı Argümanları

```bash
go run main.go [seçenekler]
```

Seçenekler:

- `-input`: Giriş ses dosyası (WAV formatı) - Dosya modu için gerekli
- `-output`: Çıkış ses dosyası (temizlenmiş) - İsteğe bağlı, varsayılan: input_cleaned.wav
- `-threshold`: Gürültü filtreleme eşik değeri (varsayılan: 200.0)
- `-verbose`: Detaylı işlem bilgisi göster
- `-streamline`: Gerçek zamanlı mikrofon gürültü azaltmayı etkinleştir

### Örnekler

1. Bir ses dosyasını işle:

```bash
go run main.go -input=input.wav -output=cleaned.wav -verbose=true
```

2. Gerçek zamanlı gürültü azaltmayı başlat:

```bash
go run main.go -streamline
```

## Çıktı

### Dosya İşleme Modu

- Temizlenmiş ses dosyası oluşturur
- İki analiz raporu oluşturur:
  1. Temizleme öncesi analiz
  2. Temizleme sonrası analiz
- Raporlar şunları içerir:
  - Ses kalite metrikleri
  - Spektral analiz
  - Gürültü analizi
  - Frekans dağılımı
  - Temizleme parametreleri

### Gerçek Zamanlı Mod

- Sesi gerçek zamanlı işler
- Durum mesajlarını gösterir
- Çıkmak için Ctrl+C tuşlarına basın

## Bağımlılıklar

- github.com/go-audio/audio
- github.com/go-audio/wav
- github.com/gordonklaus/portaudio
- gonum.org/v1/gonum/dsp/fourier

## Gereksinimler

- Go 1.21 veya daha yeni
- Sisteminizde yüklü PortAudio kütüphanesi
- Dosya işleme modu için WAV ses dosyaları
- Gerçek zamanlı mod için mikrofon ve hoparlörler

## Notlar

- Uygulama, ses analizine göre parametreleri otomatik olarak ayarlar
- Gerçek zamanlı mod, işleme nedeniyle bazı gecikmeler gösterebilir
- En iyi sonuçlar için yüksek kaliteli ses girişi kullanın
- Analiz raporları ./analyses dizinine kaydedilir
