# 🎯 Job Acceptance Prediction System

HR analytics with machine learning: predict whether a candidate will accept / be placed, find the factors
that drive it, store the cleaned data in MySQL and explore everything in an interactive Streamlit dashboard.

## What you get

| Deliverable | Where |
|---|---|
| Python scripts - EDA, preprocessing, feature engineering, ML | `src/` |
| One-command pipeline | `run_pipeline.py` |
| MySQL schema, loader, reporting queries | `sql/schema.sql`, `src/sql_loader.py`, `sql/analysis_queries.sql` |
| Streamlit dashboard (KPIs, charts, model insights, single-candidate predictor) | `app/streamlit_app.py` |
| Documentation | `docs/PROJECT_DOCUMENTATION.md` (method, data dictionary, assumptions) and `docs/RESULTS_REPORT.md` (auto-generated with *your* numbers after you run the pipeline) |

```
job_acceptance_prediction/
├── run_pipeline.py            # runs every step
├── requirements.txt
├── .env.example               # copy to .env for MySQL settings
├── data/raw/                  # <-- put HR_Job_Placement_Dataset.csv here
├── src/
│   ├── config.py              # paths, column groups, thresholds (all assumptions live here)
│   ├── data_understanding.py  # shape, dtypes, nulls, raw label problems
│   ├── preprocessing.py       # cleaning, validation, duplicates, imputation
│   ├── features.py            # engineered features
│   ├── modeling.py            # 3 models, tuning, metrics, importance, fairness, risk scores
│   ├── eda.py                 # 28 figures + EDA-guide answers
│   ├── sql_loader.py          # MySQL (or SQLite) loader
│   └── generate_report.py     # docs/RESULTS_REPORT.md
├── app/streamlit_app.py
├── sql/  docs/  tools/  .vscode/
├── reports/                   # generated: figures, metrics, quality logs
└── models/                    # generated: best_model.joblib
```

## Run it in VS Code (step by step)

**0. Prerequisites** - Python 3.10-3.13 ([python.org](https://www.python.org/downloads/); on Windows tick *Add python.exe to PATH*), VS Code with the **Python** extension. MySQL is optional (see step 6).

**1. Open the folder** - *File → Open Folder…* → select `job_acceptance_prediction`.

**2. Open a terminal** - *Terminal → New Terminal* and create a virtual environment:

```powershell
# Windows (PowerShell)
py -m venv .venv
.venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```
If PowerShell blocks activation run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.
When VS Code asks *"We noticed a new environment…"* click **Yes**; otherwise press `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `.venv`.

**3. Install packages**
```bash
pip install -r requirements.txt
```

**4. Add the data** - download `HR_Job_Placement_Dataset.csv` from the Google Drive link and save it as
`data/raw/HR_Job_Placement_Dataset.csv`.

**5. Run the pipeline**
```bash
python run_pipeline.py            # ~2-6 minutes on 50,000 rows (use --fast for a quick first test)
```
It prints each step, then creates cleaned datasets in `data/processed/`, 28 figures + metrics in `reports/`,
the trained model in `models/` and `docs/RESULTS_REPORT.md`. (Or press **F5** and pick *1. Run full pipeline* - launch configs are in `.vscode/launch.json`.)

**6. Store the cleaned data in MySQL** (optional step of the brief)
1. Install MySQL Server (or XAMPP / MySQL Workbench bundle) and make sure it is running.
2. Copy `.env.example` to `.env` and set `MYSQL_USER` / `MYSQL_PASSWORD`.
3. Run
   ```bash
   python -m src.sql_loader
   ```
   It creates the database `hr_placement`, the table `job_placement_cleaned` (typed columns from `sql/schema.sql`), inserts the rows through SQLAlchemy and runs the reporting queries in `sql/analysis_queries.sql`.
   No MySQL yet? `python -m src.sql_loader --sqlite` does the same into a local file so you can test the rest.
4. To make the dashboard read from MySQL set `DATA_SOURCE=mysql` in `.env`.

**7. Launch the dashboard**
```bash
streamlit run app/streamlit_app.py
```
It opens at <http://localhost:8501>. Use the sidebar filters; tabs: Overview · Candidate performance · Placement & acceptance · Interview & evaluation · Model insights · Predict a candidate · Data.

## Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError: Dataset not found` | Put the CSV in `data/raw/` with the exact name, or set `HR_RAW_CSV`. |
| `ModuleNotFoundError: No module named 'src'` | Run commands from the project root folder (the one containing `run_pipeline.py`). |
| `ModuleNotFoundError: sklearn / streamlit …` | The virtual environment is not active - run step 2 activation again, then `pip install -r requirements.txt`. |
| `Access denied for user` / `Can't connect to MySQL` | Check the server is running and `.env` has the right user/password. For MySQL 8 the `cryptography` package (in requirements) is required. |
| Streamlit says *Processed data not found* | Run `python run_pipeline.py` first. |
| Training feels slow | `python run_pipeline.py --fast` or `--no-tune`. |

## Assumptions you should confirm

* The **technical-score eligibility threshold** is not given in the dataset notes; **50** is assumed for the dashboard KPIs (`src/config.py` → `TECHNICAL_THRESHOLD`, or the sidebar slider). `reports/data_understanding.md` shows the placement rate per technical-score band so you can check where placement really starts.
* The data only has the *final* status, not a separate "offer made" field. Therefore: **Placement rate** = placed / all; **Job acceptance rate** = placed / eligible candidates; **Offer dropout rate** = 1 - acceptance rate; **High-risk %** = share of candidates whose predicted placement probability is below 0.30.
* `age_years` and `gender` are **not** used as model inputs (as the dataset notes recommend); a gender fairness table is produced instead.

## Testing without the real file

`python tools/make_demo_data.py --out data/raw/demo_dataset.csv` creates a **synthetic** stand-in with the same quirks
(then set `HR_RAW_CSV=data/raw/demo_dataset.csv`). Its results are invented - never present them as findings.
