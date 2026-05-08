

## ⚙️ Setup Instructions

### 1. Clone the repository

git clone https://github.com/none-commits16/fyp.git
cd fyp

### 2. Create virtual environment

python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate


### 3. Install dependencies

pip install -r requirements.txt

## 📦 Dataset Setup

> ⚠️ Data is NOT included in this repository. Download each dataset manually.

### Download DEPRESJON

1. Go to [https://datasets.simula.no/depresjon/](https://datasets.simula.no/depresjon/)
2. Download `depresjon-dataset.zip`
3. Extract it

### Download PSYKOSE

1. Go to [https://datasets.simula.no/psykose/](https://datasets.simula.no/psykose/)
2. Download all files and folders
3. Extract it

### Survey Data

Contact any team member directly — file is `data_collection.csv`


## 🗂️ Required Folder Structure

Create a `data/` folder in the project root and organise like this:

FYP/
└── data/
    ├── depresjon/
    │   ├── condition/        ← condition_1.csv ... condition_23.csv
    │   ├── control/          ← control_1.csv ... control_32.csv
    │   └── scores.csv
    ├── psykose/
    │   ├── patient/          ← patient CSV files
    │   ├── control/          ← control CSV files
    │   └── patients_info.csv
    └── data_collection.csv   ← survey data (get from team)
