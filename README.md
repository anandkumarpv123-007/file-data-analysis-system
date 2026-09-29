# File Data Analysis System

**Subtitle:** *Transform raw file-based data into meaningful insights.*  
**College Mini-Project:** Data Analysis and Essentials  
**Domain:** Data Analytics / File-Based Data Analysis  

---

## 1. Abstract

The **File Data Analysis System** is a modular, domain-agnostic web application engineered to automate the examination, quality improvement, statistical evaluation, and visual exploration of tabular datasets. 

Manual data examination across spreadsheets and raw text files is frequently tedious, error-prone, and inconsistent. This system empowers users—from students to business analysts—to upload any standard comma-, tab-, or semicolon-delimited tabular file, dynamically inspect its schema, detect and remediate data quality anomalies (such as nulls and duplicates), perform multi-criteria searches and filters, compute rigorous descriptive statistics, generate dynamic visualizations, and export publication-ready reports alongside cleaned datasets.

---

## 2. Project Objectives

The application directly implements and demonstrates the four official project objectives:

* **Objective 1 — Automation:**  
  *To automate data processing and analysis from CSV and text files, reducing the time and effort required for manual examination.*
* **Objective 2 — Data Cleaning:**  
  *To clean and improve data quality by identifying and handling missing values, duplicate records, and other data inconsistencies.*
* **Objective 3 — Data Exploration & Statistical Analysis:**  
  *To provide efficient data exploration through searching, filtering, sorting, and statistical analysis such as mean, median, minimum, and maximum.*
* **Objective 4 — Visualization & Report Generation:**  
  *To present meaningful insights clearly using summaries, tables, and visualizations to help users understand patterns, trends, and important information in the dataset.*

---

## 3. Project Modules & Architecture

The application is structured into five cohesive, decoupled modules:

```text
file-data-analysis-system/
│
├── app.py                      # Main Streamlit web application & UI navigation
├── requirements.txt            # Python dependencies
├── README.md                   # Comprehensive project documentation
│
├── data/                       # Built-in demo & testing datasets
│   ├── student_performance.csv # Primary example dataset (Education domain)
│   ├── sales_data.csv          # Multi-domain test dataset (Commercial domain)
│   ├── employee_data.csv       # Multi-domain test dataset (HR domain)
│   └── customer_data.txt       # Multi-domain test dataset (Tab-delimited text)
│
├── modules/                    # Specialized backend processing modules
│   ├── file_handler.py         # Module 1: Input ingestion, sniffer & validation
│   ├── processing.py           # Module 2: Schema inspection & dynamic typing
│   ├── cleaning.py             # Module 3: Imputation, deduplication & quality
│   ├── exploration.py          # Module 4A: Dynamic search, filter & sort
│   ├── statistics.py           # Module 4B: Numerical & categorical statistics
│   ├── visualization.py        # Module 5A: Dynamic Matplotlib & Seaborn plots
│   ├── insights.py             # Module 5B: Factual data-driven insights engine
│   └── reporting.py            # Module 5C: Text report generator & CSV exports
│
└── assets/                     # Application visual assets and documentation
```

### Module Breakdown:

1. **Module 1 — Data Input & File Handling (`modules/file_handler.py`):**
   * Upload CSV, TSV, and tabular TXT files.
   * Auto-detects delimiters (comma `,`, tab `\t`, semicolon `;`, pipe `|`) via `csv.Sniffer`.
   * Rigorous validation checking for empty files, corrupt headers, and unparseable rows.
   * Configurable dataset preview and metadata extraction.

2. **Module 2 — Data Processing (`modules/processing.py`):**
   * Dynamic column classification into Numerical, Categorical/Text, Datetime, and Boolean.
   * Memory footprint calculation and schema profiling.
   * Safe data type conversion tool (Numeric, Datetime, String, Category).

3. **Module 3 — Data Cleaning (`modules/cleaning.py`):**
   * Strict architectural preservation of original data:
     $$\text{Original Dataset} \longrightarrow \text{Working Copy} \longrightarrow \text{Cleaning} \longrightarrow \text{Cleaned Dataset}$$
   * Detection of missing values per column and duplicate records.
   * Imputation strategies: Mean, Median, Mode, Constant, or Row Dropping.
   * Whitespace trimming in text fields.
   * Before vs After verification table and audit logging.

4. **Module 4 — Data Exploration & Statistical Analysis (`modules/exploration.py` & `modules/statistics.py`):**
   * **Dynamic Search:** Substring search across all or individual text columns; operator-based searches for numerical columns.
   * **Dynamic Filtering:** Relational conditions (`==`, `!=`, `>`, `>=`, `<`, `<=`, `Between`) for numbers; text operations (`Contains`, `Starts with`, `In list`) for categories.
   * **Dynamic Sorting:** Ascending and descending ordering with null-safety.
   * **Statistical Analysis:** Single-column deep-dives (Count, Mean, Median, Min, Max, Std Dev, Variance, IQR, Skewness) and descriptive summary tables. Categorical mode and frequency breakdowns. Pearson correlation matrix.

5. **Module 5 — Data Visualization & Report Generation (`modules/visualization.py`, `modules/insights.py`, `modules/reporting.py`):**
   * Dynamic Matplotlib & Seaborn visualizations: Bar Charts, Histograms with KDE, Box Plots, Scatter Plots with regression trendlines, Line Charts, and Correlation Heatmaps.
   * Fail-safe validation preventing crashes on unsuitable dimension selections.
   * **Key Factual Insights:** 100% data-derived observations without AI hallucinations or business speculations.
   * Export of cleaned data (`cleaned_dataset.csv`), full text analysis reports (`analysis_report.txt`), and statistical summaries (`statistical_summary.csv`).

---

## 4. Technologies Used

* **Programming Language:** Python 3.10+
* **Data Processing & Analytics:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Application Framework:** Streamlit

---

## 5. Installation & Setup

### Step 1: Clone or Open the Project Folder
Open your terminal in the `file-data-analysis-system` directory:
```bash
cd file-data-analysis-system
```

### Step 2: Create and Activate a Python Virtual Environment
**On Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 6. Running the Application

Execute the Streamlit application using:
```bash
streamlit run app.py
```

Upon launch, your default browser will open automatically at:
```text
http://localhost:8501
```

---

## 7. User Workflow

The end-to-end analytical workflow follows a logical 7-step progression:

$$\text{UPLOAD} \longrightarrow \text{PROCESS} \longrightarrow \text{CLEAN} \longrightarrow \text{EXPLORE} \longrightarrow \text{ANALYZE} \longrightarrow \text{VISUALIZE} \longrightarrow \text{REPORT}$$

1. **Home:** Upload a local CSV/TXT file or click **"Example Dataset — Student Performance"** (or another demo dataset).
2. **Dataset:** View dimensions, memory size, data types, and column schema breakdown.
3. **Cleaning:** Review data health, handle nulls with mean/median/mode, remove duplicate records, and compare Before vs After.
4. **Explore:** Dynamically search keywords, apply multi-criteria filters, and sort records.
5. **Statistics:** Inspect numerical distributions, calculate mean/median/min/max/std, and analyze correlations.
6. **Visualization:** Dynamically construct charts (Bar, Histogram, Box Plot, Scatter Plot, Heatmap) and save them to the report gallery.
7. **Report:** Review factual insights and download `cleaned_dataset.csv` and `analysis_report.txt`.

---

## 8. General Dataset Compatibility & Kaggle Support

**Important Notice:**  
While **Student Performance** is provided as the primary demonstration dataset, the application is **not hard-coded** to student grades or specific column names.

The system dynamically detects any arbitrary tabular schema loaded into a Pandas DataFrame. The included demo files prove this multi-domain capability:
* **`student_performance.csv`** (Education Domain: Grades, Attendance, Study Hours)
* **`sales_data.csv`** (Commercial Domain: Orders, Discounts, Revenue, Profit)
* **`employee_data.csv`** (HR Domain: Salaries, Experience, Performance Ratings)
* **`customer_data.txt`** (Retail Domain: Tab-delimited memberships and spend)

Users can freely download any normal tabular CSV from Kaggle and upload it directly into the application without altering a single line of source code.

---

## 9. Four-Person Presentation Guide (College Viva)

To present this mini-project seamlessly as a 4-person team, the application includes a dedicated navigation guide:

* **Person 1 (Team Leader) — Data Ingestion & Processing:**
  * *Demonstrates:* Home Page $\rightarrow$ Upload / Demo Load $\rightarrow$ Dataset Overview $\rightarrow$ Dynamic Typing.
  * *Explains:* **Objective 1 (Automation)** — How automated file detection and schema profiling replaces manual file opening.
* **Person 2 — Data Quality & Cleaning:**
  * *Demonstrates:* Data Quality Health $\rightarrow$ Missing Values Table $\rightarrow$ Duplicate Rows $\rightarrow$ Imputation $\rightarrow$ Before vs After Verification.
  * *Explains:* **Objective 2 (Data Cleaning)** — The preservation of the original dataset and safe mathematical imputation strategies.
* **Person 3 — Data Exploration & Statistical Analysis:**
  * *Demonstrates:* Dynamic Search $\rightarrow$ Dynamic Filtering $\rightarrow$ Dynamic Sorting $\rightarrow$ Descriptive Statistics $\rightarrow$ Correlation.
  * *Explains:* **Objective 3 (Exploration & Stats)** — How dynamic querying and descriptive metrics adapt to any dataset.
* **Person 4 — Visualization, Insights & Report Generation:**
  * *Demonstrates:* Dynamic Matplotlib/Seaborn Plots $\rightarrow$ Validation Against Crashes $\rightarrow$ Factual Key Insights $\rightarrow$ Cleaned Dataset Export.
  * *Explains:* **Objective 4 (Visualization & Insights)** — Translating raw records into graphical evidence and reproducible deliverables.

---

## 10. License & Credits

Developed as part of the **College Mini-Project in Data Analysis and Essentials**.
Built with Python, Pandas, NumPy, Matplotlib, Seaborn, and Streamlit.
