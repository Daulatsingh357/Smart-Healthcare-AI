# SmartCare AI - AI-Powered Smart Healthcare System

Flask + MySQL/SQLite + scikit-learn/XGBoost + Chart.js + Bootstrap 5. B.Tech CSE final-year project.
Your existing app (auth, doctors, appointments, medicines, admin) is kept; the five AI modules are integrated into it, share its database, login and patient dashboard, and are reachable from one unified dashboard.

> **Medical safety.** This is an educational project. It never diagnoses, prescribes or gives doses. Every result is labelled as an estimate, and emergency symptoms always trigger "call 112/108" guidance.

## 1. What you get

| # | Module | URL | Core tech | Main tables |
|---|--------|-----|-----------|-------------|
| 1 | Multi-disease risk prediction (diabetes, heart, kidney, liver) | `/predict` | LR, RF, Decision Tree, SVM, XGBoost, calibrated; accuracy/precision/recall/F1/ROC-AUC/confusion matrix; what-if explanations; PDF | `disease_predictions`, `prediction_inputs`, `model_versions` |
| 2 | Medical report analyzer | `/reports` | PDF text + Tesseract OCR, regex parser, **ranges only from the report**, EN/HI explanations, encrypted storage, comparison charts, PDF | `medical_reports`, `extracted_report_values` |
| 3 | AI health assistant | `/chatbot` | Safety rules + TF-IDF retrieval (EN/HI), context memory, consent-gated personal data, voice in/out | `chatbot_conversations`, `chatbot_messages` |
| 4 | Diet & nutrition planner | `/diet` | BMI/BMR (Mifflin-St Jeor), safe targets, allergen/condition filtering, content-based ranking + kNN swaps, Indian food DB (94 foods), trackers, PDF | `diet_plans`, `diet_meals`, `nutrition_tracking` |
| 5 | Predictive health analytics | `/analytics` | Linear regression / Random Forest (walk-forward chosen), Isolation Forest, rolling mean, **gated forecasts with 95 % intervals**, monthly summary, PDF | `health_metrics`, `health_analytics` |
| 6 | Unified dashboard | `/dashboard` | Live data from all five modules | - |

UI: blue/white theme, sidebar, animated cards, Chart.js, toasts, loaders, **dark mode**, **English/Hindi toggle** (top bar).

## 2. Honest limitations (read before your viva)

* **Training data is synthetic by default.** No real patient dataset could be downloaded in the build environment, so each disease model trains on generated data whose labels come from a hand-written formula. The metrics (about 0.66-0.78 accuracy) show the pipeline works; they are **not** clinical evidence. The UI says so on every prediction. To use real data, see section 7.
* Low/Moderate/High cut-offs (30 %/60 %) are **illustrative**, not clinically validated.
* **The chatbot is not an LLM.** It is a deterministic safety layer + TF-IDF retrieval over a curated bilingual knowledge base (20 topics + 28 lab terms). It answers what is in that knowledge base and otherwise says it does not know. Extend `ml_models/chat_kb.py` to add topics.
* **OCR** works best on clear, straight scans of "name  value  unit  range" layouts. Multi-column or handwritten reports may not parse. The raw extracted text is shown on each report so you can check quality. Tests with no printed range are shown as **Not assessed**, never guessed.
* PDF exports are **English only** (ReportLab cannot shape Devanagari without extra fonts). Hindi is available on screen for the UI, chatbot and report explanations. UI translation covers navigation, headings, main buttons and disclaimers, not every sentence.
* Voice input uses the browser Web Speech API (Chrome/Edge); text-to-speech uses the browser's voices (a Hindi voice must be installed in your OS).
* Forecasts need **at least 6 readings on different days spanning 14+ days**; the Isolation Forest needs 15+ multi-metric readings. Below that the app says why it will not forecast.
* Food values are approximate typical-serving figures (IFCT/NIN style), not laboratory analyses.
* Verified here: all 89 automated tests on SQLite **and** on MariaDB 10.11 (MySQL-compatible), `schema.sql` applied to MariaDB, real Tesseract OCR on a generated image, all page JavaScript syntax-checked with Node. **Not verified here:** behaviour in a real browser (charts, voice, dark mode), Windows, and real MySQL 8. Do a quick click-through on your machine.

## 3. Windows CMD installation

Prerequisites: Python 3.10-3.12 (tick "Add to PATH"), optional MySQL 8, [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki) (needed only for JPG/PNG/scanned PDFs; text-based PDFs work without it).

```cmd
cd smart_healthcare
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Edit `.env`: set a long random `SECRET_KEY`, change `ADMIN_PASSWORD`, and (if Tesseract is not on PATH) `TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe`.

**Database** - choose one:
* *SQLite (default, nothing to do).*
* *MySQL:* 
```cmd
mysql -u root -p -e "CREATE DATABASE smart_healthcare CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
```
then in `.env`: `DATABASE_URL=mysql+pymysql://root:YOUR_PASSWORD@localhost/smart_healthcare?charset=utf8mb4`. Tables are created automatically on first start; `database\schema.sql` is the equivalent hand-runnable DDL (`mysql -u root -p < database\schema.sql`).

**Train the models** (about 1 minute; writes `ml_models\saved_models\` and `docs\MODEL_EVALUATION.md`):
```cmd
python ml_models\train_all.py
```
**Optional demo data** (one patient with 75 days of fictional history so every module has something to show):
```cmd
python scripts\seed_demo_data.py
```
**Run:**
```cmd
python app.py
```
Open http://127.0.0.1:5000. Demo login `demo@patient.local` / `Demo@12345`; admin from `.env`; demo doctors `dr.general@demo.local`, `dr.heart@demo.local` (same password). 

**Tests:**
```cmd
python -m pytest -q
set TEST_DATABASE_URL=mysql+pymysql://user:pw@127.0.0.1/hc_test?charset=utf8mb4
python -m pytest -q
```
(The second run uses MySQL; the test database is emptied after every test, so use a dedicated schema.)

Upgrading your **existing** database: new columns are added automatically at start-up (`utils/migrate.py`) without touching data. The old `predictions` and `health_records` tables are no longer used and can be dropped.

## 4. How to demonstrate each module (about 10 minutes)

1. **Dashboard** - log in as the demo patient: five module cards, risk history chart, latest reports, nutrition, weight progress, recommendations.
2. **Prediction** - `/predict` > Heart tab > Analyze. Show the ring, the three "what-if" factors, the *synthetic data* warning, the PDF. Change age/BP/smoker and re-run to show the risk move. Open *Model evaluation* for the five algorithms and confusion matrices.
3. **Reports** - upload `database\sample_data\sample_lab_report.pdf` (fictional) or your own lab PDF/JPG. Show extracted vs AI-explained sections, red out-of-range rows, "Not assessed" for tests with no printed range, switch to हिं for Hindi explanations, the comparison chart and latest-vs-previous table, the PDF, and that `uploads\*.enc` is unreadable on disk.
4. **Chatbot** - ask "How can I lower my cholesterol?", then "what about exercise?" (context), "मुझे कितना पानी पीना चाहिए?" (Hindi), "I have chest pain" (emergency), "which tablet for fever and what dose" (refusal), turn on *Use my health history* and ask "explain my report". Use the mic button and *Read aloud*.
5. **Diet** - Planner (try weight loss with BMI 17 to see the safety block, or add an allergy), Weekly plan (swap a food = kNN alternatives), Calorie calculator, Tracker (log food, water, weight; watch the dashboard update).
6. **Analytics** - log a reading with high BP (alert), show trend charts with the dashed estimate and 95 % band, anomaly table, risk trend, monthly summary, *Run & save analysis*, PDF. Register a new patient to show "no forecast: need 6 readings over 14 days".

## 5. REST API (JSON, session cookie + `X-CSRFToken` header)

| Method & path | Purpose |
|---|---|
| `POST /api/predictions` `{disease, inputs, algorithm?}` | run + store a prediction (422 with per-field errors) |
| `GET /api/predictions[?disease=]`, `GET/DELETE /api/predictions/<id>`, `GET .../<id>/pdf`, `GET /api/models` | history, detail, PDF, model metrics |
| `POST /api/reports` (multipart `file,title,report_date`) | upload + OCR + extract |
| `GET /api/reports`, `/summary`, `/parameters`, `/compare?param=`, `/compare/latest`, `/<id>`, `/<id>/pdf`, `/<id>/file`; `DELETE /<id>` | dashboard, comparison, downloads |
| `POST /api/chat` `{message, conversation_id?, language}`; `GET /api/chat/conversations[/<id>]`; `DELETE`; `POST /api/chat/consent` | chat |
| `POST /api/diet/metrics`, `POST /api/diet/plan`, `GET /api/diet/plans[/<id>[/pdf]]`, `POST /api/diet/plans/<id>/swap`, `GET /api/diet/foods`, `/foods/<id>/alternatives`, `POST /api/diet/calc` | planner |
| `GET /api/nutrition/summary`, `POST /api/nutrition/log\|water\|weight`, `DELETE /api/nutrition/<id>` | tracking |
| `POST/GET /api/metrics`, `DELETE /api/metrics/<id>`, `GET /api/analytics/overview\|monthly\|history\|report.pdf`, `POST /api/analytics/run` | analytics |

All `/api/*` routes return 401 JSON when logged out and 403 when the record belongs to another patient.

## 6. Security measures

Password hashing (Werkzeug), CSRF protection on every state-changing request, role checks, per-record ownership checks (tested), upload size limit + magic-byte validation + extension match, uploaded files encrypted with Fernet (AES) under a random filename, secure cookie flags, `nosniff`/frame/referrer headers, input range validation, parameterised SQL (SQLAlchemy), output escaping in templates and JS, consent switch for chatbot access to health history, cascading deletion of all patient data when an admin removes a user. For real deployment add HTTPS, rate limiting, audit logging, backups and a legal/consent review.

## 7. Using real datasets

Put a CSV at `database\datasets\<disease>.csv` (`diabetes`, `heart`, `kidney`, `liver`) with the columns listed in `ml_models\<disease>_model.py` (`SPEC["features"]`) plus a 0/1 `target` column, then run `python ml_models\train_all.py`. The data-source line in the UI, in `model_versions` and in the PDFs then changes from SYNTHETIC to your file. Commonly used public sources are the Pima Indians Diabetes data, the UCI Heart Disease (Cleveland) data, the UCI Chronic Kidney Disease data and the Indian Liver Patient Dataset; their columns differ from this app's features, so map or adapt `SPEC["features"]` accordingly and **check each licence and cite it in your report**. `database\sample_data\synthetic_*.csv` shows the exact expected format (synthetic).

## 8. Project layout

```
app.py  config.py  .env  requirements.txt  pytest.ini
models/__init__.py              all SQLAlchemy models (18 tables)
routes/                         auth, patient (dashboard), appointments, admin,
                                prediction, report_analyzer, chatbot, diet_planner, health_analytics
ml_models/                      common.py (engine), diabetes|heart|kidney|liver_model.py, train_all.py,
                                report_extractor.py, chat_engine.py + chat_kb.py, diet_recommender.py, health_analyzer.py
utils/                          pdf.py, security.py (encryption), migrate.py, model_registry.py, context.py
templates/ static/              Jinja pages, CSS, JS (app.js helpers, i18n.js)
database/                       schema.sql, indian_foods.csv, sample_data/, datasets/ (your real CSVs)
scripts/                        seed_demo_data.py, generate_sample_dataset.py, export_schema.py
tests/                          89 pytest tests          docs/  documentation + model evaluation
```

## 9. Troubleshooting

* *"Models are not trained yet"* - run `python ml_models\train_all.py`.
* *OCR error / "Tesseract is not installed"* - install Tesseract and set `TESSERACT_CMD` in `.env`. Text-based PDFs never need it.
* *Uploaded reports unreadable after changing `SECRET_KEY`* - set a fixed `FILE_ENCRYPTION_KEY` once and never change it.
* *MySQL "Access denied" / driver errors* - check the `DATABASE_URL` user/password and that `PyMySQL` is installed.
* *Charts missing* - the page loads Bootstrap/Chart.js from a CDN, so you need internet access (or vendor those files into `static\`).
