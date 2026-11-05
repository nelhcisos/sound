"""Скачать бесплатные модели-образцы (Cults3D) с Яндекс.Диска пользователя и распаковать в models/cults/<архив>/.
usage: python3 download_yadisk.py [публичная_ссылка]
По умолчанию — папка «3dtrain» (6 моделей, ~1,25 ГБ, после распаковки ~2,4 ГБ). Папка models/ в .gitignore.
"""
import sys, os, json, urllib.request, urllib.parse, zipfile, hashlib
LINK = sys.argv[1] if len(sys.argv) > 1 else "https://disk.yandex.com/d/VcN3AwnpAHnn6g"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "cults")
os.makedirs(OUT, exist_ok=True)
api = "https://cloud-api.yandex.net/v1/disk/public/resources?limit=200&public_key=" + urllib.parse.quote(LINK)
items = json.load(urllib.request.urlopen(api))["_embedded"]["items"]
for it in items:
    if it["type"] != "file": continue
    dst = os.path.join(OUT, it["name"])
    if not (os.path.exists(dst) and os.path.getsize(dst) == it["size"]):
        print("качаю", it["name"], round(it["size"] / 1e6), "МБ")
        for attempt in range(4):
            try:
                urllib.request.urlretrieve(it["file"], dst); break
            except Exception as e:
                print("  повтор:", e)
    md5 = hashlib.md5(open(dst, "rb").read()).hexdigest()
    print(it["name"], "md5", "ok" if md5 == it["md5"] else "НЕ СОВПАЛ")
    if dst.endswith(".zip"):
        d = dst[:-4]
        if not os.path.isdir(d):
            zipfile.ZipFile(dst).extractall(d)
print("готово ->", OUT)
