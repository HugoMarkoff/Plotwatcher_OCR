
# 📸 Plotwatcher Camera Trap OCR & Timelapse Analyzer

This project provides an **OCR-based extractor and analyzer** for **Plotwatcher camera trap images**.  
It automatically detects timestamps, dates, temperatures, and battery levels from images, and reconstructs **timelapse sequences** even when OCR fails.  

If you need to convert timelapse (TLV) formats look at:  https://github.com/HeyHarry3636/TimeLapseVideo_python 

✅ Tested on **Ubuntu 24.04** with **Python 3.12**  
⚡ Supports **GPU acceleration** with **PyTorch (CUDA)** and **PaddleOCR**  
🛠️ Falls back to **EasyOCR** if PaddleOCR is unavailable   (Actually defaults to it right now)

---

## 🚀 Features

- **OCR extraction** of:
  - Date & time (from image overlay or filename)
  - Temperature (°C/°F, with OCR error correction)
  - Battery level (%)
- **Timelapse sequence analysis**:
  - Detects sequence numbers and intervals
  - Predicts missing timestamps
  - Handles sequence resets across days
- **Post-processing corrections**:
  - Interpolates missing temperatures
  - Corrects outlier values (temperature spikes, impossible battery jumps)
  - Fills missing times using sequence-based interpolation
- **Exports results to CSV** for further analysis
- **Debug mode** with detailed OCR and prediction logs

---

## 🖥️ Setup

### 1. Clone the repository

git clone https://github.com/HugoMarkoff/Plotwatcher_OCR.git

cd Plotwatcher_OCR

### 2. Create a virtual environment (Python 3.12)

python3.12 -m venv venv
source venv/bin/activate

### 3. Install dependencies

pip install --upgrade pip
pip install -r requirements.txt

---

## ⚡ GPU / CUDA Notes

This project uses **PyTorch** and **PaddleOCR**.  
The `requirements.txt` is pinned for **CUDA 12.1** on Ubuntu 24.04.  

If your system has a **different CUDA version**, you must adjust the PyTorch and PaddlePaddle installs.

### Example: Install PyTorch for CUDA 11.8

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

### Example: Install CPU-only PyTorch

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

### Example: Install PaddlePaddle for CPU only

pip install paddlepaddle

👉 **Tip:** Check your CUDA version with:

nvidia-smi

---

## ▶️ Usage

### Process a folder of images

python plotwatcher.py /path/to/images --output results.csv

### Enable debug mode (detailed OCR + predictions)

python plotwatcher.py /path/to/images --debug

### Use PaddleOCR instead of EasyOCR

python plotwatcher.py /path/to/images --paddleOCR

---

## 📂 Input Folder Structure

The tool supports **nested folders**.  
Example:

/data/plotwatcher/
├── cam1/day1/200731AA_000095.jpg
├── cam1/day2/200801AA_000245.jpg
└── cam2/200922AA_001048.jpg

You can point the script at `/data/plotwatcher/` and it will automatically recurse into all subfolders.

---

## 📊 Output

The script produces a **CSV file** with columns:

- `filename` – original image filename
- `Date` – extracted or interpolated timestamp
- `Temperature_C` – corrected temperature in °C
- `Battery_Level` – corrected battery percentage
- `filename_date` – date parsed from filename
- `sequence_number` – timelapse sequence number
- `sequence_identifier` – optional ID (if present)
- `frame_id` – frame identifier (if not timelapse)
- `has_timelapse` – whether the file is part of a timelapse
- `filename_format` – detected filename format

---

## 🔧 Post-Processing Logic

- **Temperature correction**
  - Removes outliers (e.g. sudden 40°C jumps)
  - Interpolates missing values from neighbors
- **Battery correction**
  - Prevents impossible increases
  - Limits drop rate (max 1% per 100 frames)
  - Fills missing values with gradual decline
- **Time correction**
  - Uses OCR when available
  - Predicts missing times from sequence intervals
  - Handles day rollovers and sequence resets

---

## 📈 Example Debug Output

🔤 PLOTWATCHER PROCESSED: 200922AA_001048.jpg

   📝 OCR Texts Found: ['PRO', '09:21:26', '52%']
   
   📅 Filename Date: 2020-09-22
   
   🔢 Sequence Number: 001048
   
   ⏰ OCR Time Found: 09:21:26 ✅
   
   ⏱️  Sequence Interval: 10s [cached]
   
   🌡️  Temperature: No temperature found
   
   🔋 Battery Level: 52%

---

## ✅ Tested Environment

- **OS:** Ubuntu 24.04 LTS  
- **Python:** 3.12  
- **CUDA:** 12.1  
- **PyTorch:** 2.3.1+cu121  
- **PaddleOCR:** 2.7.0.3  
- **EasyOCR:** 1.7.1  

---

## 📜 License

MIT License — free to use and modify.






