from functools import wraps
from flask import abort, jsonify
from flask_login import current_user, login_required


def roles_required(*roles):
    def deco(f):
        @wraps(f)
        @login_required
        def w(*a, **k):
            if current_user.role not in roles:
                abort(403)
            return f(*a, **k)
        return w
    return deco


def api_error(message, code=400, **extra):
    return jsonify(error=message, **extra), code


def current_patient():
    return current_user.patient


def fnum(v):
    try:
        if v is None or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None
