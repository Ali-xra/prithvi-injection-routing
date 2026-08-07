# -*- coding: utf-8 -*-
"""نسخهٔ تیرهٔ THE-GATE-COMPLETE برای خواندن روی صفحه.

محتوا از فایل روشن می‌آید؛ اینجا فقط شیوه‌نامه و رنگ‌های داخل SVG عوض می‌شود،
به‌علاوهٔ چند انیمیشن که در نسخهٔ روشن نبود.
"""
import io, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

D = r"C:\Users\aliso\Desktop\proje\uni\ideas\prithvi-injection-routing\docs"
t = open(D + r"\THE-GATE-COMPLETE.html", encoding="utf-8").read()

# ── ۱ · رنگ‌های داخل SVG: روشن → تیره ────────────────────────────────
MAP = [
 # جعبه‌ها
 ("#eef2f7", "#1d2732"), ("#e9f0fa", "#1d2732"), ("#eef4f8", "#1b2530"),
 ("#dce7f2", "#22303e"), ("#fdf1e2", "#32281a"), ("#f5efe4", "#2d2920"),
 ("#eef7ef", "#18291e"), ("#e4f0e6", "#1a2a1f"), ("#fdf2f2", "#2c1d1d"),
 ("#f3f3f0", "#21242a"), ("#f2f2ef", "#21242a"), ("#f4f4f1", "#21242a"),
 ("#e8e8e8", "#282c33"), ("#f2f9f3", "#18291e"),
 # خط‌ها
 ("#9fb0c4", "#5f83ad"), ("#7f9ec4", "#6f9ecd"), ("#9fb8cc", "#6890ac"),
 ("#8fa8c8", "#5f83ad"), ("#d8a86a", "#b58547"), ("#c9ab7c", "#8d7853"),
 ("#8fbf9c", "#4d9268"), ("#dcaaaa", "#a45f5f"), ("#bbb", "#484d56"),
 ("#999", "#5c626b"), ("#ccc", "#484d56"),
 # متن‌ها
 ("#33465e", "#9dc0e0"), ("#2c4a6e", "#9dc0e0"), ("#1c3f5c", "#9dc8e8"),
 ("#6a7c92", "#8ea2b7"), ("#5b7898", "#8ea2b7"), ("#22562f", "#7fd6a0"),
 ("#4b7a57", "#6fbc8c"), ("#8a5a17", "#e0b070"), ("#6b5124", "#d8b184"),
 ("#8a2020", "#ee9090"), ("#1d6b3a", "#6fd69a"), ("#2f7d4f", "#6fd69a"),
 ("#b03030", "#e88080"), ("#c0392b", "#e08070"), ("#5a6470", "#98a2ae"),
 ("#9aa2ad", "#98a2ae"),
 ("#111", "#eceff2"), ("#222", "#dce1e6"), ("#333", "#cfd6dd"),
 ("#444", "#c3cad2"), ("#555", "#aab3bd"), ("#666", "#98a2ae"),
 ("#888", "#8b95a1"), ("#4a7fb5", "#6fa8dc"),
]
# 🔴 باگِ نسخهٔ قبل: با partition("</style>") سرِ فایل شاملِ خودِ CSS روشن می‌ماند
#    و یک <style> تودرتو ساخته می‌شد، پس شیوه‌نامهٔ تیره اصلاً اعمال نمی‌شد.
#    حالا کلِ بلوکِ style پیدا و کنار گذاشته می‌شود.
m = re.search(r"<style>[\s\S]*?</style>", t)
head = t[:m.start()]
body = t[m.end():]
for a, b in MAP:
    body = body.replace('"' + a + '"', '"' + b + '"')
# چهار عکسِ هم‌میانگین: روی زمینهٔ تیره باید برعکس شوند
body = body.replace('"#1a1a1a" : "#e9e9e9"', '"#e8e8e8" : "#363b43"')

# ── ۲ · شیوه‌نامهٔ تیره ──────────────────────────────────────────────
CSS = """
 :root{--bg:#14161a;--panel:#1b1e24;--line:#2a2f38;--tx:#dfe4ea;--dim:#9aa4b0;
       --teal:#4ec9a0;--amber:#e0a458;--red:#e07878;--blue:#7fb0e0}
 *{scrollbar-color:#3a4049 var(--bg)}
 body{font-family:"Segoe UI",Tahoma,sans-serif;line-height:1.95;max-width:50em;
      margin:0 auto;padding:2.4em 2em 6em;color:var(--tx);background:var(--bg);
      font-size:16.5px}
 svg{direction:ltr}
 h1{font-size:1.85em;color:#fff;border-bottom:2px solid var(--teal);
    padding-bottom:.32em;margin-bottom:.15em;letter-spacing:.2px}
 h2{font-size:1.28em;margin-top:3em;margin-bottom:.6em;color:#fff;
    border-right:4px solid var(--teal);padding-right:.7em}
 h3{font-size:1.05em;margin-top:1.9em;margin-bottom:.4em;color:#c9d2db}
 .sub{color:var(--dim);font-size:.92em;margin-bottom:2.4em}
 p{margin:.95em 0;text-align:justify}
 .key,.good,.bad,.warn{padding:1em 1.25em;margin:1.4em 0;background:var(--panel);
      border:1px solid var(--line);border-right-width:5px;border-radius:0 5px 5px 0;
      box-shadow:0 1px 3px rgba(0,0,0,.35)}
 .key {border-right-color:var(--blue)}
 .good{border-right-color:var(--teal)}
 .bad {border-right-color:var(--red)}
 .warn{border-right-color:var(--amber)}
 .fig{border:1px solid var(--line);background:#171a1f;padding:1.15em .8em .9em;
      margin:1.9em 0;border-radius:6px;box-shadow:0 2px 10px rgba(0,0,0,.4)}
 .cap{font-size:.9em;color:var(--dim);text-align:center;margin-top:.8em;
      line-height:1.85;border-top:1px solid var(--line);padding-top:.65em}
 .btn{background:#252a32;color:#dfe4ea;border:1px solid #39404a;border-radius:5px;
      padding:.46em 1.15em;font-family:inherit;font-size:.9em;cursor:pointer;
      margin:.25em;transition:all .18s}
 .btn:hover{background:#2f3540;border-color:#4a525e;transform:translateY(-1px)}
 .btn.on{background:#1d5c42;border-color:var(--teal);color:#c8f0dd}
 table{border-collapse:collapse;width:100%;margin:1.3em 0;font-size:.95em}
 th,td{border:1px solid var(--line);padding:.55em .75em;text-align:right;
       vertical-align:top}
 th{background:#232830;color:#fff}
 tr:nth-child(even) td{background:#191c21}
 td.n{font-family:Consolas,monospace;direction:ltr;text-align:left;color:#bfe6d4}
 code{font-family:Consolas,monospace;direction:ltr;font-size:.92em;
      background:#232830;color:#9fd8bd;padding:.1em .38em;border-radius:3px}
 pre{background:#0f1114;color:#d6dbe0;border:1px solid var(--line);padding:1em 1.2em;
     overflow-x:auto;direction:ltr;text-align:left;font-size:13px;line-height:1.75;
     border-radius:6px}
 pre .c{color:#6f8f78}
 b{color:#fff}
 /* ظاهرشدن نرمِ بخش‌ها هنگام اسکرول — فقط وقتی JS روشن است.
    کلاس js روی <html> با اسکریپت گذاشته می‌شود؛ اگر JS نبود، هیچ‌چیز مخفی نمی‌ماند. */
 html.js h2,html.js .fig,html.js .key,html.js .good,html.js .bad,html.js .warn,
 html.js table,html.js pre{opacity:0;transform:translateY(14px);
   transition:opacity .5s ease,transform .5s ease}
 html.js .seen{opacity:1 !important;transform:none !important}
 @media (prefers-reduced-motion:reduce){
   html.js h2,html.js .fig,html.js .key,html.js .good,html.js .bad,html.js .warn,
   html.js table,html.js pre{opacity:1;transform:none}}
 @media print{body{background:#fff;color:#000;max-width:none}
   html.js h2,html.js .fig,html.js .key,html.js .good,html.js .bad,html.js .warn,
   html.js table,html.js pre{opacity:1 !important;transform:none !important}}
"""
t = head + "<style>" + CSS + "</style>" + body
assert t.count("<style>") == 1, "بیش از یک بلوک style — دوباره همان باگ"

# ── ۳ · ظاهرشدن هنگام اسکرول ───────────────────────────────────────
REVEAL = """
<script>
(function(){
  if (!("IntersectionObserver" in window)) return;   /* بدون آن، هیچ‌چیز مخفی نشود */
  document.documentElement.classList.add("js");
  var io = new IntersectionObserver(function(es){
    for (var i = 0; i < es.length; i++)
      if (es[i].isIntersecting) es[i].target.classList.add("seen");
  }, {threshold: .12});
  var sel = "h2,.fig,.key,.good,.bad,.warn,table,pre";
  var n = document.querySelectorAll(sel);
  for (var i = 0; i < n.length; i++) io.observe(n[i]);
  setTimeout(function(){ for (var i = 0; i < 6 && i < n.length; i++)
    n[i].classList.add("seen"); }, 120);
})();
</script>
"""
t = t.replace("</body>", REVEAL + "\n</body>")

open(D + r"\THE-GATE-COMPLETE-DARK.html", "w", encoding="utf-8", newline="\n").write(t)
print("ok DARK", len(t))
print("light leftovers:", len(re.findall(r'"#[ef][0-9a-f]{5}"', t)))
