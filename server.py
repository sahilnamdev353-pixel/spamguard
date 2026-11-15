import io, os, re, secrets, shutil, sqlite3
from datetime import datetime, timedelta
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
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax", MAX_CONTENT_LENGTH=6 * 1024 * 1024)

model = joblib.load(os.path.join(BASE, "spam_model.joblib"))
vec, nb = model.steps[0][1], model.steps[-1][1]
TOKEN = re.compile(r"(?u)\b\w\w+\b")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


try:
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = 40_000_000
except ImportError:
    Image = None

MAX_UPLOAD = 5 * 1024 * 1024
KINDS = ("email", "sms", "screenshot", "pdf")
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

class OcrUnavailable(Exception):
    pass

def tesseract_path():
    for c in (os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
              r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe")):
        if c and os.path.exists(c):
            return c
    return None

def ocr(img):
    try:
        import pytesseract
    except ImportError:
        raise OcrUnavailable("Reading screenshots needs pytesseract. Run: python -m pip install pytesseract")
    path = tesseract_path()
    if not path:
        raise OcrUnavailable("Reading screenshots needs Tesseract OCR, which was not found on this computer.")
    pytesseract.pytesseract.tesseract_cmd = path
    return pytesseract.image_to_string(img.convert("RGB"), lang="eng", timeout=40)

def looks_like_image(b):
    return (b[:8] == b"\x89PNG\r\n\x1a\n" or b[:3] == b"\xff\xd8\xff"
            or (b[:4] == b"RIFF" and b[8:12] == b"WEBP") or b[:2] == b"BM")

def extract_image(data):
    if Image is None:
        raise OcrUnavailable("Reading screenshots needs Pillow. Run: python -m pip install Pillow")
    return ocr(Image.open(io.BytesIO(data)))

def extract_pdf(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise OcrUnavailable("Reading PDFs needs pypdf. Run: python -m pip install pypdf")
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("encrypted PDF")
    text = "\n".join((p.extract_text() or "") for p in reader.pages[:5])
    if len(text.strip()) >= 30:
        return text
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise OcrUnavailable("This PDF is a scan. Reading scanned PDFs needs pypdfium2. Run: python -m pip install pypdfium2")
    pdf = pdfium.PdfDocument(data)
    return "\n".join(ocr(pdf[i].render(scale=2).to_pil()) for i in range(min(len(pdf), 5)))

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

def run_check(text, kind):
    p, parts = analyze(text)
    conf = p if p >= 0.5 else 1 - p
    db().execute("INSERT INTO checks(user_id,kind,content,is_spam,confidence,created_at) VALUES(?,?,?,?,?,?)",
                 (g.user["id"], kind, text, int(p >= 0.5), conf, now()))
    db().commit()
    return dict(spam=p >= 0.5, conf=round(conf * 100, 1), pct=round(p * 100, 1), parts=parts, kind=kind)

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

STATS = "SELECT COUNT(*) c, COALESCE(SUM(is_spam),0) s FROM checks WHERE user_id=?"

@app.route("/dashboard")
@login_required
def dashboard():
    uid = g.user["id"]
    t = db().execute(STATS, (uid,)).fetchone()
    total, spam = t["c"], t["s"]
    days = [datetime.now().date() - timedelta(days=i) for i in range(6, -1, -1)]
    rows = db().execute("SELECT substr(created_at,1,10) d, COUNT(*) c, COALESCE(SUM(is_spam),0) s FROM checks "
                        "WHERE user_id=? AND created_at >= ? GROUP BY d", (uid, days[0].strftime("%Y-%m-%d"))).fetchall()
    m = {r["d"]: r for r in rows}
    chart = []
    for d in days:
        r = m.get(d.strftime("%Y-%m-%d"))
        c, sp = (r["c"], r["s"]) if r else (0, 0)
        chart.append(dict(label=d.strftime("%a"), day=d.strftime("%d %b"), spam=sp, ham=c - sp, total=c))
    top = max([c["total"] for c in chart] + [1])
    recent = db().execute("SELECT * FROM checks WHERE user_id=? ORDER BY id DESC LIMIT 5", (uid,)).fetchall()
    return render_template("dashboard.html", total=total, spam=spam, ham=total - spam,
                           rate=round(spam * 100 / total) if total else 0, chart=chart, top=top, recent=recent)

@app.route("/check", methods=["GET", "POST"])
@login_required
def check():
    result = None
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        kind = request.form.get("kind")
        kind = kind if kind in KINDS else "email"
        if not text:
            flash("Paste a message first.")
        elif len(text) > 5000:
            flash("Message is too long. Keep it under 5,000 characters.")
        else:
            result = run_check(text, kind)
    return render_template("check.html", result=result)

@app.errorhandler(413)
def too_large(e):
    flash("File is too large. The maximum size is 5 MB.")
    return redirect(url_for("check"))

@app.route("/upload", methods=["POST"])
@login_required
def upload():
    f = request.files.get("file")
    if not f or not f.filename:
        flash("Choose a screenshot or PDF first.")
        return redirect(url_for("check"))
    ext = os.path.splitext(f.filename.lower())[1]
    data = f.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        flash("File is too large. The maximum size is 5 MB.")
        return redirect(url_for("check"))
    try:
        if ext == ".pdf" and data[:5] == b"%PDF-":
            text, kind = extract_pdf(data), "pdf"
        elif ext in IMG_EXT and looks_like_image(data):
            text, kind = extract_image(data), "screenshot"
        else:
            flash("Unsupported file. Upload a PNG, JPG, WEBP or PDF.")
            return redirect(url_for("check"))
    except OcrUnavailable as e:
        flash(str(e))
        return redirect(url_for("check"))
    except Exception:
        flash("Could not read that file. Try a clearer screenshot or paste the text.")
        return redirect(url_for("check"))
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\x00", "")).strip()[:5000]
    if not text:
        flash("No readable text was found in that file.")
        return redirect(url_for("check"))
    result = run_check(text, kind)
    flash("File read and checked. If the extracted text has mistakes, fix it below and press Check message again.")
    return render_template("check.html", result=result, prefill=text, kind=kind)

@app.route("/history")
@login_required
def history():
    f = request.args.get("f", "")
    q, args = "SELECT * FROM checks WHERE user_id=?", [g.user["id"]]
    if f in ("spam", "ham"):
        q += " AND is_spam=?"
        args.append(1 if f == "spam" else 0)
    rows = db().execute(q + " ORDER BY id DESC LIMIT 200", args).fetchall()
    return render_template("history.html", rows=rows, f=f)

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    uid = g.user["id"]
    if request.method == "POST":
        act = request.form.get("action")
        if act == "name":
            name = request.form.get("name", "").strip()
            if not name or len(name) > 60:
                flash("Enter your name (up to 60 characters).")
            else:
                db().execute("UPDATE users SET name=? WHERE id=?", (name, uid))
                db().commit()
                flash("Name updated.")
                return redirect(url_for("profile"))
        elif act == "password":
            u = db().execute("SELECT password_hash FROM users WHERE id=?", (uid,)).fetchone()
            cur, new, conf = (request.form.get(k, "") for k in ("current", "new", "confirm"))
            if not check_password_hash(u["password_hash"], cur):
                flash("Current password is incorrect.")
            elif len(new) < 8:
                flash("New password must be at least 8 characters.")
            elif new != conf:
                flash("The two new passwords do not match.")
            else:
                db().execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(new), uid))
                db().commit()
                flash("Password changed.")
                return redirect(url_for("profile"))
    u = db().execute("SELECT name, email, created_at FROM users WHERE id=?", (uid,)).fetchone()
    t = db().execute(STATS, (uid,)).fetchone()
    return render_template("profile.html", u=u, total=t["c"], spam=t["s"])

@app.route("/history/<int:cid>/delete", methods=["POST"])
@login_required
def delete_check(cid):
    db().execute("DELETE FROM checks WHERE id=? AND user_id=?", (cid, g.user["id"]))
    db().commit()
    return redirect(url_for("history"))

if __name__ == "__main__":
    init_db()
    app.run(host="127.0.0.1", port=5000, debug=False)
