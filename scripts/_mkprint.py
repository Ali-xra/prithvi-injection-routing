# -*- coding: utf-8 -*-
"""نسخهٔ چاپیِ THE-GATE-COMPLETE را می‌سازد.

دکمه‌ها حذف می‌شوند و همهٔ شکل‌ها در لحظهٔ باز شدن صفحه به حالت نهایی می‌روند،
چون چند شکل با JS ساخته می‌شوند و بدون آن خالی چاپ می‌شدند.
"""
import io, re, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SRC = r"C:\Users\aliso\Desktop\proje\uni\ideas\prithvi-injection-routing\docs\THE-GATE-COMPLETE.html"
DST = r"C:\Users\aliso\Desktop\proje\uni\ideas\prithvi-injection-routing\docs\THE-GATE-COMPLETE-PRINT.html"

t = open(SRC, encoding="utf-8").read()

# ۱ · دکمه‌ها بیرون
t = re.sub(r'<div style="text-align:center">\s*(<button[\s\S]*?</button>\s*)+</div>', "", t)
t = re.sub(r"<button[\s\S]*?</button>", "", t)

# ۲ · شیوه‌نامهٔ چاپ
PRINT_CSS = """
 body{font-family:"Segoe UI",Tahoma,sans-serif;line-height:1.72;max-width:none;
      margin:0;padding:0;color:#000;background:#fff;font-size:11.5pt}
 svg{direction:ltr}
 h1{font-size:19pt;color:#000;border-bottom:1.5pt solid #000;padding-bottom:4pt;
    margin:0 0 3pt}
 h2{font-size:13.5pt;margin:20pt 0 6pt;color:#000;border-right:3pt solid #000;
    padding-right:7pt;page-break-after:avoid}
 h3{font-size:11.8pt;margin:12pt 0 3pt;color:#000}
 .sub{color:#444;font-size:10pt;margin-bottom:14pt}
 p{margin:6pt 0;text-align:justify}
 .key,.good,.bad,.warn{padding:7pt 10pt;margin:9pt 0;border:0.6pt solid #999;
      border-right-width:3pt;page-break-inside:avoid;background:#fff}
 .key {border-right-color:#000}
 .good{border-right-color:#1c6b3f}
 .bad {border-right-color:#a02020}
 .warn{border-right-color:#8a6010}
 .fig{border:0.6pt solid #aaa;background:#fff;padding:8pt 6pt;margin:12pt 0;
      page-break-inside:avoid}
 .cap{font-size:9.5pt;color:#333;text-align:center;margin-top:5pt;line-height:1.6;
      border-top:0.4pt solid #ccc;padding-top:4pt}
 table{border-collapse:collapse;width:100%;margin:8pt 0;font-size:10pt;
       page-break-inside:avoid}
 th,td{border:0.5pt solid #888;padding:4pt 6pt;text-align:right;vertical-align:top}
 th{background:#e8e8e8;color:#000}
 td.n{font-family:Consolas,monospace;direction:ltr;text-align:left}
 code{font-family:Consolas,monospace;direction:ltr;font-size:9.8pt;
      background:#f0f0f0;padding:0 2pt}
 pre{background:#f4f4f4;color:#000;border:0.5pt solid #bbb;padding:7pt 9pt;
     direction:ltr;text-align:left;font-size:9.5pt;line-height:1.55;
     page-break-inside:avoid;white-space:pre-wrap}
 pre .c{color:#4a6a4a;font-style:italic}
 b{color:#000}
 @page{size:A4;margin:16mm 15mm}
"""
t = re.sub(r"<style>[\s\S]*?</style>", "<style>" + PRINT_CSS + "</style>", t, count=1)

# ۳ · یادداشت بالای صفحه
NOTE = ('<div style="border:1.2pt solid #000;padding:7pt 10pt;margin:0 0 14pt;'
        'font-size:10pt;line-height:1.6">'
        '<b>نسخهٔ چاپی.</b> همان محتوای نسخهٔ صفحه‌ای، ولی همهٔ شکل‌ها در حالتِ '
        'نهایی‌شان چاپ می‌شوند و دکمه‌ای وجود ندارد. '
        'برای نسخهٔ تعاملی: <code>THE-GATE-COMPLETE.html</code></div>')
i = t.find("<body>")
t = t[:i + 6] + "\n" + NOTE + t[i + 6:]

# ۴ · همه‌چیز در لحظهٔ بارگذاری به حالت نهایی
AUTORUN = """
<script>
window.addEventListener("load", function(){
  var f = ["grow","stepGo","proofGo","meanGo","scatGo","depthGo"];
  for (var i = 0; i < f.length; i++) { try { window[f[i]](); } catch(e){} }
  var all = document.querySelectorAll("[opacity]");
  setTimeout(function(){
    for (var j = 0; j < all.length; j++){
      var v = all[j].getAttribute("opacity");
      if (v !== null && parseFloat(v) < 1) all[j].setAttribute("opacity", 1);
    }
    try {
      document.getElementById("b1").setAttribute("width", 470);
      document.getElementById("b2").setAttribute("width", 470 * 0.4 / 21.3);
    } catch(e){}
  }, 4200);
});
</script>
"""
t = t.replace("</body>", AUTORUN + "\n</body>")

open(DST, "w", encoding="utf-8", newline="\n").write(t)
print("ok", DST, len(t), "chars")
print("buttons left:", t.count("<button"))
