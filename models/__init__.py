"""SQLAlchemy models. All patient-owned rows carry patient_id (FK -> patients.id) and are indexed."""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy.dialects import mysql

db = SQLAlchemy()
LongText = db.Text().with_variant(mysql.LONGTEXT(), "mysql")


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(10), nullable=False, index=True)  # admin / doctor / patient
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Patient(db.Model):
    __tablename__ = "patients"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    chat_context_consent = db.Column(db.Boolean, default=False)  # consent to use health history in chatbot
    user = db.relationship("User", backref=db.backref("patient", uselist=False))

    @property
    def bmi(self):
        if self.height_cm and self.weight_kg:
            return round(self.weight_kg / ((self.height_cm / 100) ** 2), 1)
        return None


class Doctor(db.Model):
    __tablename__ = "doctors"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), unique=True, nullable=False)
    specialization = db.Column(db.String(80), index=True)
    experience_years = db.Column(db.Integer, default=0)
    bio = db.Column(db.Text)
    user = db.relationship("User", backref=db.backref("doctor", uselist=False))


class Appointment(db.Model):
    __tablename__ = "appointments"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey("doctors.id"), nullable=False, index=True)
    scheduled_at = db.Column(db.DateTime, nullable=False)
    reason = db.Column(db.String(255))
    status = db.Column(db.String(12), default="Pending")  # Pending/Confirmed/Completed/Cancelled
    patient = db.relationship("Patient")
    doctor = db.relationship("Doctor")


class Medicine(db.Model):
    __tablename__ = "medicines"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), index=True)
    name = db.Column(db.String(100))
    dosage = db.Column(db.String(60))
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)
    reminder_time = db.Column(db.String(5))


class Feedback(db.Model):
    __tablename__ = "feedback"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    message = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship("User")


# ---------------------------------------------------------------- Module 1: prediction
class ModelVersion(db.Model):
    __tablename__ = "model_versions"
    id = db.Column(db.Integer, primary_key=True)
    disease = db.Column(db.String(20), nullable=False, index=True)
    algorithm = db.Column(db.String(40), nullable=False)
    version = db.Column(db.String(30), nullable=False, index=True)
    data_source = db.Column(db.String(255))
    n_samples = db.Column(db.Integer)
    accuracy = db.Column(db.Float)
    precision_score = db.Column(db.Float)
    recall = db.Column(db.Float)
    f1 = db.Column(db.Float)
    roc_auc = db.Column(db.Float)
    confusion_matrix = db.Column(db.JSON)   # [[tn, fp], [fn, tp]]
    is_best = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True, index=True)
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)


class DiseasePrediction(db.Model):
    __tablename__ = "disease_predictions"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    disease = db.Column(db.String(20), nullable=False, index=True)
    model_name = db.Column(db.String(40))
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_versions.id"), nullable=True)
    risk_percent = db.Column(db.Float, nullable=False)
    risk_level = db.Column(db.String(10), nullable=False)   # Low / Moderate / High
    top_factors = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    inputs = db.relationship("PredictionInput", backref="prediction", cascade="all, delete-orphan")
    model_version = db.relationship("ModelVersion")


class PredictionInput(db.Model):
    __tablename__ = "prediction_inputs"
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("disease_predictions.id", ondelete="CASCADE"), nullable=False, index=True)
    feature_name = db.Column(db.String(40), nullable=False)
    feature_value = db.Column(db.Float, nullable=False)


# ---------------------------------------------------------------- Module 2: reports
class MedicalReport(db.Model):
    __tablename__ = "medical_reports"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), index=True)
    title = db.Column(db.String(120))
    filename = db.Column(db.String(200))          # random name of the ENCRYPTED file on disk
    original_name = db.Column(db.String(200))
    file_type = db.Column(db.String(10))          # pdf / png / jpg
    size_bytes = db.Column(db.Integer)
    sha256 = db.Column(db.String(64))
    report_date = db.Column(db.Date, index=True)
    ocr_text = db.Column(LongText)
    extraction_method = db.Column(db.String(30))  # pdf-text / ocr / pdf-image-ocr
    status = db.Column(db.String(20), default="analyzed")  # analyzed / no_values
    value_count = db.Column(db.Integer, default=0)
    abnormal_count = db.Column(db.Integer, default=0)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    values = db.relationship("ExtractedReportValue", backref="report", cascade="all, delete-orphan",
                             order_by="ExtractedReportValue.id")


class ExtractedReportValue(db.Model):
    __tablename__ = "extracted_report_values"
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey("medical_reports.id", ondelete="CASCADE"), nullable=False, index=True)
    param_key = db.Column(db.String(40), nullable=False, index=True)
    param_name = db.Column(db.String(80))          # name exactly as printed in the report
    value = db.Column(db.Float, nullable=False)
    unit = db.Column(db.String(30))
    ref_low = db.Column(db.Float)
    ref_high = db.Column(db.Float)
    ref_text = db.Column(db.String(60))            # reference range text exactly as printed
    status = db.Column(db.String(20))              # Low / Normal / High / Not assessed
    status_source = db.Column(db.String(20))       # report-range / report-flag / none
    raw_line = db.Column(db.String(255))


# ---------------------------------------------------------------- Module 3: chatbot
class ChatConversation(db.Model):
    __tablename__ = "chatbot_conversations"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    title = db.Column(db.String(120))
    language = db.Column(db.String(5), default="en")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, index=True)
    messages = db.relationship("ChatMessage", backref="conversation", cascade="all, delete-orphan",
                               order_by="ChatMessage.id")


class ChatMessage(db.Model):
    __tablename__ = "chatbot_messages"
    id = db.Column(db.Integer, primary_key=True)
    conversation_id = db.Column(db.Integer, db.ForeignKey("chatbot_conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = db.Column(db.String(10), nullable=False)  # user / assistant
    content = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(5))
    is_emergency = db.Column(db.Boolean, default=False)
    meta = db.Column(db.JSON)                        # topic, followups, level, doctors
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


# ---------------------------------------------------------------- Module 4: diet
class DietPlan(db.Model):
    __tablename__ = "diet_plans"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    goal = db.Column(db.String(20))
    bmi = db.Column(db.Float)
    bmr = db.Column(db.Float)
    tdee = db.Column(db.Float)
    calories = db.Column(db.Integer)
    protein_g = db.Column(db.Integer)
    carbs_g = db.Column(db.Integer)
    fat_g = db.Column(db.Integer)
    water_ml = db.Column(db.Integer)
    inputs = db.Column(db.JSON)
    warnings = db.Column(db.JSON)
    is_active = db.Column(db.Boolean, default=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    meals = db.relationship("DietMeal", backref="plan", cascade="all, delete-orphan",
                            order_by="(DietMeal.day_index, DietMeal.id)")


class DietMeal(db.Model):
    __tablename__ = "diet_meals"
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey("diet_plans.id", ondelete="CASCADE"), nullable=False, index=True)
    day_index = db.Column(db.Integer, nullable=False)    # 0 = Monday
    meal_type = db.Column(db.String(12), nullable=False)  # breakfast / lunch / snack / dinner
    items = db.Column(db.JSON)
    kcal = db.Column(db.Float)
    protein_g = db.Column(db.Float)
    carbs_g = db.Column(db.Float)
    fat_g = db.Column(db.Float)


class NutritionTracking(db.Model):
    __tablename__ = "nutrition_tracking"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    log_date = db.Column(db.Date, nullable=False, index=True)
    entry_type = db.Column(db.String(8), nullable=False)  # food / water
    meal_type = db.Column(db.String(12))
    food_id = db.Column(db.Integer)
    food_name = db.Column(db.String(100))
    servings = db.Column(db.Float)
    kcal = db.Column(db.Float, default=0)
    protein_g = db.Column(db.Float, default=0)
    carbs_g = db.Column(db.Float, default=0)
    fat_g = db.Column(db.Float, default=0)
    water_ml = db.Column(db.Integer, default=0)
    logged_at = db.Column(db.DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------- Module 5: analytics
class HealthMetric(db.Model):
    __tablename__ = "health_metrics"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False, index=True)
    systolic = db.Column(db.Float)
    diastolic = db.Column(db.Float)
    blood_sugar = db.Column(db.Float)
    heart_rate = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    cholesterol = db.Column(db.Float)
    source = db.Column(db.String(20), default="manual")   # manual / prediction / nutrition
    __table_args__ = (db.Index("ix_health_metrics_patient_time", "patient_id", "recorded_at"),)


class HealthAnalytics(db.Model):
    __tablename__ = "health_analytics"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False, index=True)
    analysis_type = db.Column(db.String(30), nullable=False, index=True)  # snapshot / monthly_summary
    period_start = db.Column(db.Date)
    period_end = db.Column(db.Date)
    result = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)


# Patient-owned tables with a direct patient_id (used when an admin deletes a user).
PATIENT_OWNED = [Appointment, Medicine, DiseasePrediction, MedicalReport, ChatConversation, DietPlan,
                 NutritionTracking, HealthMetric, HealthAnalytics]
