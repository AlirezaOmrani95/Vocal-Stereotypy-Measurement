# Vocal-Stereotypy-Measurement
This repository contains the implementation code for the paper:  
**"Machine Learning to Measure Vocal Stereotypy: An Extension"**

The goal of this project is to detect and measure vocal stereotypy using a machine learning model trained on relevant audio features.
---

## 🧪 How to Run the Testing Pipeline

There are **two ways** to run the testing pipeline:

### Option 1: 🖥️ Use the Windows Software (Recommended for Non-Developers)
We provide an installable Windows application that allows you to run the model without setting up the code environment.

- ✅ No coding or Python setup required
- 📥 Download the software from: [Zenodo link](https://zenodo.org/records/13284337)
- 🧾 Installation instructions are included in the download package

---

### Option 2: 🧑‍💻 Run the Code Manually (For Developers or Custom Use)

To run the testing code directly:

1. **Clone this repository**  
   ```bash
   git clone https://github.com/AlirezaOmrani95/Vocal-Stereotypy-Measurement.git
   cd Vocal-Stereotypy-Measurement

2. **Install dependencies**
   ```bash
   install -r requirements.txt

3. **Download the pretrained model weights from** [Zenodo](https://zenodo.org/records/13284337)

4. **Create the correct folder structure**
    Based on the expected inputs to the `arg_parsing` function in cli.py.

5. **Run the test script**
   ```bash
   python test.py --best_weight_dir ./weight/weights.pth --threshold 0.5 --dataset_dir ./dataset

## Project Structure
```bash
Vocal-Stereotypy-Measurement/
├── utils/               # Utility modules (audio, data, general helpers)
│   ├── audio.py
│   ├── data_utils.py
│   ├── general.py
│   └── __init__.py
├── cli.py               # Command-line interface and argument parsing
├── train.py             # Training script
├── test.py              # Testing script
├── pretrained_model.py  # Model loader
├── states.py            # State logic
├── constants.py         # Global constants
├── scripts/             # (Reserved for helper scripts)
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── LICENSE              # License information
```
## License
This project is licensed under the **[MIT License](LICENSE)**.
You are free to use, modify, and distribute this software under the terms of the license.

## Citation
If you use this code or software in your research, please cite:
```bibtex
@misc{omrani_lanovaz_moroni_2024,
 title={Machine Learning to Detect Vocal Stereotypy: Improving Duration-Based Measures},
 url={osf.io/preprints/psyarxiv/c4k98_v1},
 DOI={10.31234/osf.io/c4k98},
 publisher={PsyArXiv},
 author={Omrani, Ali R and Lanovaz, Marc J and Moroni, Davide},
 year={2024},
 month={Aug}
}
