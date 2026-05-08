1. Clone the repository
bashgit clone https://github.com/YOURUSERNAME/mental-health-monitoring-fyp.git
cd mental-health-monitoring-fyp
2. Create virtual environment
bashpython -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt

How to download DEPRESJON

Go to (https://datasets.simula.no/depresjon/)
Download depresjon-dataset.zip
Extract it

How to download PSYKOSE

Go to (https://datasets.simula.no/psykose/)
Download all files and folders
Extract it

Survey data
Contact any team member — file is data_collection.csv
Required folder structure
Create a data/ folder in the project root and organise like this:
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
    └── data_collection.csv   ← survey data
