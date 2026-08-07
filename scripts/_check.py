# -*- coding: utf-8 -*-
import sys, io, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

P = r"C:\Users\aliso\Desktop\proje\uni\ideas\prithvi-injection-routing\docs\THE-GATE-COMPLETE.html"
t = open(P, encoding="utf-8").read()

print("chars", len(t), " lines", t.count("\n"))
print("mojibake:", ("\u00d8" in t or "\u00d9" in t))
print("script:", t.count("<script"), t.count("</script>"))
print("svg   :", t.count("<svg"), t.count("</svg>"))
print("backtick template literals:", t.count("`"))

ids_used = set(re.findall(r'getElementById\("([A-Za-z0-9_]+)"\)', t))
ids_have = set(re.findall(r'id="([A-Za-z0-9_]+)"', t))
miss = sorted(i for i in ids_used if i not in ids_have)
print("ids referenced but not defined:", miss)

fns = set(re.findall(r'function\s+([A-Za-z0-9_]+)\s*\(', t))
calls = set(re.findall(r'onclick="([A-Za-z0-9_]+)\(', t))
print("onclick handlers missing:", sorted(c for c in calls if c not in fns))
