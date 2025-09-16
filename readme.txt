SB01 Data Visualization App

Install packages in your VS Code virtual environment:

1. Activate your virtual environment (venv or conda):
   For macOS/Linux:
     source venv/bin/activate
   For Windows (PowerShell):
     venv\Scripts\activate

2. Install required packages:
   pip install pandas numpy plotly streamlit openpyxl

Required packages:
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- plotly: Interactive plotting
- streamlit: Web app framework
- openpyxl: Excel file reader

Run the application:
streamlit run test_up.py
