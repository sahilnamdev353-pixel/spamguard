import os, re, secrets, sqlite3
from datetime import datetime
from functools import wraps
import joblib
from flask import Flask, g, render_template, request, redirect, url_for, session, flash, abort
from werkzeug.security import generate_password_hash, check_password_hash

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "spamguard.db")
KEY_FILE = os.path.join(BASE, ".secret_key")
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "w") as f:
        f.write(secrets.token_hex(32))

app = Flask(__name__)
app.secret_key = open(KEY_FILE).read().strip()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", MAX_CONTENT_LENGTH=1024 * 1024)

model = joblib.load(os.path.join(BASE, "spam_model.joblib"))
vec, nb = model.steps[0][1], model.steps[-1][1]
TOKEN = re.compile(r"(?u)\b\w\w+\b")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(e=None):
    d = g.pop("db", None)
    if d:
        d.close()

def init_db():
    with sqlite3.connect(DB) as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS checks(id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id),
            kind TEXT NOT NULL, content TEXT NOT NULL, is_spam INTEGER NOT NULL,
            confidence REAL NOT NULL, created_at TEXT NOT NULL);""")

def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    return session["csrf"]
app.jinja_env.globals["csrf_token"] = csrf_token

@app.before_request
def before():
    if request.method == "POST" and not secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "")):
        abort(400)
    g.user = None
    if "uid" in session:
        g.user = db().execute("SELECT id, name, email FROM users WHERE id=?", (session["uid"],)).fetchone()

def login_required(f):
    @wraps(f)
    def wrap(*a, **k):
        if not g.user:
            flash("Log in to continue.")
            return redirect(url_for("login"))
        return f(*a, **k)
    return wrap

def analyze(text):
    p = float(model.predict_proba([text])[0][1])
    diff = nb.feature_log_prob_[1] - nb.feature_log_prob_[0]
    parts, pos = [], 0
    for m in TOKEN.finditer(text):
        i = vec.vocabulary_.get(m.group().lower())
        parts.append((text[pos:m.start()], False))
        parts.append((m.group(), p >= 0.5 and i is not None and diff[i] > 1.5))
        pos = m.end()
    parts.append((text[pos:], False))
    return p, parts

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw, pw2 = request.form.get("password", ""), request.form.get("confirm", "")
        if not name or len(name) > 60:
            flash("Enter your name (up to 60 characters).")
        elif not EMAIL.match(email):
            flash("Enter a valid email address.")
        elif len(pw) < 8:
            flash("Password must be at least 8 characters.")
        elif pw != pw2:
            flash("The two passwords do not match.")
        else:
            try:
                db().execute("INSERT INTO users(name,email,password_hash,created_at) VALUES(?,?,?,?)",
                             (name, email, generate_password_hash(pw), now()))
                db().commit()
                flash("Account created. Log in to start checking messages.")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                flash("An account with this email already exists.")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        u = db().execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if u and check_password_hash(u["password_hash"], request.form.get("password", "")):
            session.clear()
            session["uid"] = u["id"]
            return redirect(url_for("dashboard"))
        flash("Email or password is incorrect.")
    return render_template("login.html")

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You are logged out.")
    return redirect(url_for("index"))

@app.route("/check", methods=["GET", "POST"])
@login_required
def dashboard():
    result = None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        kind = "sms" if request.form.get("kind") == "sms" else "email"
        if not text:
            flash("Paste a message first.")
        elif len(text) > 5000:
            flash("Message is too long. Keep it under 5,000 characters.")
        else:
            p, parts = analyze(text)
            conf = p if p >= 0.5 else 1 - p
            db().execute("INSERT INTO checks(user_id,kind,content,is_spam,confidence,created_at) VALUES(?,?,?,?,?,?)",
                         (g.user["id"], kind, text, int(p >= 0.5), conf, now()))
            db().commit()
            result = dict(spam=p >= 0.5, conf=round(conf * 100, 1), parts=parts, kind=kind)
    n = db().execute("SELECT COUNT(*) c, COALESCE(SUM(is_spam),0) s FROM checks WHERE user_id=?", (g.user["id"],)).fetchone()
    return render_template("dashboard.html", result=result, total=n["c"], spam=n["s"])

@app.route("/history")
@login_required
def history():
    rows = db().execute("SELECT * FROM checks WHERE user_id=? ORDER BY id DESC LIMIT 200", (g.user["id"],)).fetchall()
    return render_template("history.html", rows=rows)

@app.route("/history/<int:cid>/delete", methods=["POST"])
@login_required
def delete_check(cid):
    db().execute("DELETE FROM checks WHERE id=? AND user_id=?", (cid, g.user["id"]))
    db().commit()
    return redirect(url_for("history"))

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False)
