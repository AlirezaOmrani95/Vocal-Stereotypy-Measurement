# Vocal-Stereotypy-Measurement
This repository contains the implementation code for the paper:  
**"Machine Learning to Detect Vocal Stereotypy: Improving Duration-Based Measures"**

**The goal of this project is to detect and measure vocal stereotypy using a machine learning model trained on relevant audio features.**

You can find the paper [here](https://journals.sagepub.com/doi/full/10.1177/01454455251380510).

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

## Disclaimer

This tool is for research and educational purposes only. It is not intended for clinical use.

## Citation
If you use this code or software in your research, please cite:
```bibtex
@article{doi:10.1177/01454455251380510,
   author = {Ali Reza Omrani and Marc J. Lanovaz and Davide Moroni},
   title ={Machine Learning to Detect Vocal Stereotypy: Improving Duration-Based Measures},
   journal = {Behavior Modification},
   volume = {0},
   number = {0},
   pages = {01454455251380510},
   year = {0},
   doi = {10.1177/01454455251380510},
       note ={PMID: 41103133},
   URL = {https://doi.org/10.1177/01454455251380510},
   eprint = {https://doi.org/10.1177/01454455251380510}
}
