# AI-Powered Smart Healthcare System - Project Documentation

## 1. Problem and scope
Patients need one place to assess risk, understand lab reports, ask health questions, plan nutrition and watch long-term trends. This project extends an existing Flask healthcare portal (authentication, doctors, appointments, medicine reminders, admin panel) with five AI modules and a unified dashboard. It is an **educational decision-support prototype**, not a medical device.

## 2. Architecture
```
Browser (Bootstrap 5, Chart.js, vanilla JS, Web Speech API)
   | JSON over HTTPS + CSRF token                    | HTML (Jinja)
Flask app factory (app.py) --- blueprints: auth, patient, appointments, admin,
   |                                         prediction, report_analyzer, chatbot, diet_planner, health_analytics
   |-- utils/: pdf (ReportLab), security (Fernet), context (cross-module queries), migrate, model_registry
   |-- ml_models/: engines (no Flask imports; unit-testable)
   '-- SQLAlchemy -> MySQL 8 / SQLite     files: uploads/*.enc (encrypted)     models: ml_models/saved_models/*.joblib
```
Design principles: thin routes, ML code independent of Flask, every patient-owned row has `patient_id` with an ownership check on every access, safety logic implemented as code (not prompt text) and covered by tests.

## 3. Database (18 tables; `database/schema.sql`)
Existing: `users, patients, doctors, appointments, medicines, feedback`.
Required new tables: `disease_predictions` (+`prediction_inputs`, FK `model_version_id`), `model_versions`, `medical_reports`, `extracted_report_values`, `chatbot_conversations`, `chatbot_messages`, `diet_plans`, `diet_meals`, `nutrition_tracking`, `health_metrics`, `health_analytics`.
Relationships: patient 1-N predictions / reports / conversations / diet plans / metrics / analytics / nutrition rows; prediction 1-N inputs; report 1-N extracted values; conversation 1-N messages; plan 1-N meals. Child rows are deleted by ORM cascade; FKs and indexes (patient_id, timestamps, disease, parameter key) are defined on all tables; JSON columns hold factors, meal items and analysis snapshots; `LONGTEXT` holds OCR text on MySQL.

## 4. Module designs
### 4.1 Multi-disease prediction
Feature sets per disease (age, gender, BMI/height/weight, BP, glucose, cholesterol, heart rate and disease-specific labs) are validated against plausible ranges; out-of-range input is rejected, never clipped. Five algorithms (Logistic Regression, Random Forest, Decision Tree, SVM, XGBoost) are each wrapped in `StandardScaler` + sigmoid-calibrated `CalibratedClassifierCV` so that outputs behave as probabilities. A stratified 75/25 split yields accuracy, precision, recall, F1, ROC-AUC and the confusion matrix; the best ROC-AUC model is deployed and every version is recorded in `model_versions`. Explanations use a one-feature-at-a-time substitution against the training median ("what-if"), which is simple, honest and model-agnostic. Risk bands: <30 % Low, 30-60 % Moderate, >=60 % High (illustrative).
**Data:** synthetic by default (formula-labelled). Real CSVs in `database/datasets/` are picked up automatically; the data source is stored with the model and printed on every result and PDF.

### 4.2 Report analyzer
Pipeline: magic-byte validation -> text (PDF text layer; else OCR of embedded page images; images via Tesseract with grey-scale/auto-contrast/upscaling) -> line parser (longest-alias match of 28 known tests, first number = value, unit from a unit list, range from `a-b`, `<x`, `>x`) -> status computed **only** from the range printed in the report (or an H/L flag printed in the report). A test without a printed range becomes "Not assessed". Stored: encrypted file, raw text, each value with the original line so users can verify. Explanations (English and Hindi) come from a developer-written dictionary and are visibly tagged "AI-generated"; extracted data are tagged "EXTRACTED". Historical comparison charts each parameter across reports and shows latest-vs-previous deltas (flagging unit mismatches).

### 4.3 Chatbot
Order of processing per message: language detection -> **emergency rules** (chest pain, breathing difficulty, stroke signs, unconsciousness, seizures, heavy bleeding, poisoning, anaphylaxis, self-harm; English and Hindi; negation-aware) -> **refusal rules** (diagnosis, prescriptions, doses) -> small talk -> consent-gated personal-data intents -> TF-IDF char-n-gram retrieval with a word-overlap guard (blocks off-topic matches) and a lab-term alias lookup -> follow-up handling from the previous topic -> "see a doctor soon" escalation on duration/worsening cues, with doctors from the portal's own `doctors` table. Suggested follow-up questions are the canonical questions of related topics (tested to be answerable). Conversations and flags are persisted.

### 4.4 Diet planner
Mifflin-St Jeor BMR x activity factor = maintenance; loss deficit = min(20 %, 500 kcal); gain surplus = min(12 %, 400 kcal); floors 1500/1200 kcal; protein by goal (kidney 0.8 g/kg), fat 25-28 %, carbohydrate remainder (>=130 g); water 35 ml/kg, withheld for kidney/heart conditions. Blocked: age <18, pregnancy, loss when BMI <18.5 or target BMI <18.5, gain when BMI >=25. Planning: filter foods by preference, allergens (with synonyms) and condition flags (high-GI, high-sodium, fried, high saturated fat); choose items per meal slot by cosine similarity of macro shares to the target (recency penalty for variety); scale portions to the slot's calories and fine-tune each day to within about 4-5 % of target; report weekly adequacy. Swaps use k-nearest neighbours on standardised nutrient vectors, re-checked against the restrictions. Trackers store food, water and weight; dashboards compare with targets.

### 4.5 Predictive analytics
Per metric: descriptive trend (OLS slope/week, R^2), 3-point rolling mean, and a **gated** 7-day forecast (>=6 readings on different days over >=14 days). A linear model always competes with a lag-feature Random Forest (only if >=30 observed days with few gaps) in a 7-step walk-forward test; the lower-error model is used and the interval is the OLS prediction interval or 1.96 x walk-forward RMSE. Anomalies: Isolation Forest over the patient's own multivariate history (>=15 readings), explained by the metric with the largest robust z-score; fixed reference-threshold alerts always run. Risk-trend monitoring applies the same gating to the prediction history. Preventive suggestions are transparent rules combining predictions, BMI, trends, lab abnormalities, anomalies and nutrition logs. A monthly summary compares with the previous month and can be saved and exported.

## 5. Testing
`python -m pytest -q` runs 89 tests (about 30 s): 
* prediction (5 algorithms present, metrics present, validation, BMI derivation, ownership, PDF, versions), 
* report analyzer (no invented ranges, encryption at rest, fake-file rejection, duplicates, comparison, isolation, real OCR on an image), 
* chatbot (emergency/self-harm/negation/refusal, bilingual, every canonical question answerable, consent gating, history isolation), 
* diet (formulas, calorie floors, unsafe-request blocks, preference/allergy/condition constraints, swaps, tracking, isolation), 
* analytics (forecast gating, intervals, trend, Isolation Forest, threshold alerts, API validation, snapshots, PDF), 
* dashboard/admin (all modules reflected, cascade delete, additive DB migration from the old schema), 
* auth/security (login required, role checks, headers, all pages render, required tables exist).
The same suite passes against MariaDB 10.11 (`TEST_DATABASE_URL`). Page scripts were syntax-checked with Node; browser behaviour was not tested automatically.

## 6. Limitations and future work
Synthetic training data; unvalidated risk bands; knowledge-base chatbot (an LLM with retrieval and the same safety layer in front would be the next step); OCR limited to line-style tables; English-only PDFs; approximate food values; no clinician review workflow; no rate limiting/audit log. Future: real datasets with external validation and calibration plots, wearable-data import, FHIR export, doctor-side patient view with consent, push reminders.

## 7. Viva talking points
Why calibrate probabilities? Why not impute reference ranges? Why gate forecasts? What does an anomaly mean vs a diagnosis? How are emergencies detected before any retrieval? Why synthetic data and what would change with real data (class balance, leakage, external validation)? How is data protected (encryption at rest, ownership checks, consent)? How does the kNN swap differ from the ranking step?
