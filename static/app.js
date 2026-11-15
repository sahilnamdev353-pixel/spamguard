(function () {
  var root = document.documentElement, btn = document.getElementById("theme");
  function dark() { var t = root.dataset.theme; return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; }
  function label() { if (btn) btn.textContent = dark() ? "Light" : "Dark"; }
  label();
  if (btn) btn.addEventListener("click", function () {
    var n = dark() ? "light" : "dark"; root.dataset.theme = n;
    try { localStorage.setItem("theme", n); } catch (e) {}
    label();
  });
  document.querySelectorAll("[data-gauge]").forEach(function (g) {
    var pct = parseFloat(g.dataset.gauge), fg = g.querySelector(".g-fg"), num = g.querySelector(".g-num");
    fg.style.strokeDasharray = "0 100";
    requestAnimationFrame(function () { requestAnimationFrame(function () { fg.style.strokeDasharray = pct + " 100"; }); });
    var start = null;
    function step(ts) { start = start || ts; var k = Math.min((ts - start) / 900, 1); num.textContent = Math.round(pct * k) + "%"; if (k < 1) requestAnimationFrame(step); }
    requestAnimationFrame(step);
  });
})();
