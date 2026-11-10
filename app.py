import os
from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_login import LoginManager, current_user
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash
from config import Config
from models import db, User


def create_app(overrides=None):
    app = Flask(__name__)
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    if app.config["SECRET_KEY"].startswith("dev-only") and not app.config.get("TESTING"):
        app.logger.warning("SECRET_KEY is the development default. Set a long random value in .env before deploying.")
    db.init_app(app)
    CSRFProtect(app)
    lm = LoginManager(app)
    lm.login_view = "auth.login"

    @lm.user_loader
    def load(uid):
        return db.session.get(User, int(uid))

    @lm.unauthorized_handler
    def unauthorized():
        if request.path.startswith("/api/"):
            return jsonify(error="Please log in."), 401
        return redirect(url_for("auth.login"))

    from ml_models import common, report_extractor
    common.MODEL_DIR = app.config["MODEL_DIR"]
    report_extractor.configure_tesseract(app.config.get("TESSERACT_CMD"))
    from routes.auth import bp as auth
    from routes.patient import bp as patient
    from routes.appointments import bp as appts
    from routes.admin import bp as admin
    from routes.prediction import bp as prediction
    from routes.report_analyzer import bp as reports
    from routes.chatbot import bp as chatbot
    from routes.diet_planner import bp as diet
    from routes.health_analytics import bp as analytics
    for b in (auth, patient, appts, admin, prediction, reports, chatbot, diet, analytics):
        app.register_blueprint(b)

    @app.route("/")
    def index():
        return redirect(url_for("patient.dashboard")) if current_user.is_authenticated else render_template("index.html")

    @app.after_request
    def headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    def _err(code, msg):
        if request.path.startswith("/api/"):
            return jsonify(error=msg), code
        return render_template("error.html", code=code, message=msg), code

    for code, msg in ((403, "You do not have access to this resource."), (404, "Not found."), (413, "File too large (max 8 MB)."), (400, "Bad request.")):
        app.register_error_handler(code, lambda e, c=code, m=msg: _err(c, getattr(e, "description", None) if c == 400 and getattr(e, "description", None) else m))

    with app.app_context():
        db.create_all()
        from utils.migrate import ensure_columns
        ensure_columns(db)
        from utils.model_registry import register_models
        try:
            register_models()
        except Exception as exc:  # metrics.json missing/corrupt must not stop the app
            app.logger.warning("Model registry sync skipped: %s", exc)
        email = os.getenv("ADMIN_EMAIL", "admin@healthcare.local")
        if not app.config.get("TESTING") and not User.query.filter_by(email=email).first():
            db.session.add(User(name="Admin", email=email, role="admin", password_hash=generate_password_hash(os.getenv("ADMIN_PASSWORD", "Admin@12345"))))
            db.session.commit()
    return app


if __name__ == "__main__":
    create_app().run(debug=os.getenv("FLASK_DEBUG", "1") == "1")
