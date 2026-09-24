"""青空文庫から著作権切れ・新字新仮名の作品を集めて、学習用コーパスを作る。

  python3 data/fetch.py [目標文字数]

出力: data/works/<作品ID>.txt（作品ごと）, data/corpus.txt（結合、作品の間は空行 2 つ）
"""
import csv, io, re, sys, zipfile, concurrent.futures, urllib.request, pathlib, time, random

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 6_000_000
ROOT = pathlib.Path(__file__).resolve().parent
WORKS = ROOT / "works"
WORKS.mkdir(exist_ok=True)

AUTHORS = [
    "夏目 漱石", "芥川 竜之介", "太宰 治", "宮沢 賢治", "森 鴎外", "中島 敦", "梶井 基次郎", "坂口 安吾",
    "江戸川 乱歩", "夢野 久作", "島崎 藤村", "有島 武郎", "横光 利一", "堀 辰雄", "国木田 独歩", "新美 南吉",
    "小川 未明", "海野 十三", "岡本 綺堂", "吉川 英治", "織田 作之助", "山本 周五郎", "久生 十蘭", "豊島 与志雄",
    "田中 貢太郎", "菊池 寛", "小林 多喜二", "葉山 嘉樹", "萩原 朔太郎", "石川 啄木", "徳田 秋声", "泉 鏡花",
    "寺田 寅彦", "中谷 宇吉郎", "折口 信夫", "柳田 国男", "北原 白秋", "室生 犀星", "牧野 信一", "林 芙美子",
    "宮本 百合子", "伊藤 左千夫", "二葉亭 四迷", "尾崎 紅葉", "幸田 露伴", "正岡 子規", "高村 光太郎", "中原 中也",
    "レイモンド チャンドラー", "ドストエフスキー フィヨードル・ミハイロヴィチ", "ポー エドガー・アラン",
]

def load_index():
    with open(ROOT / "list_person_all_extended_utf8.csv", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    keep = []
    for r in rows:
        if r["文字遣い種別"] != "新字新仮名": continue
        if r["作品著作権フラグ"] != "なし" or r["人物著作権フラグ"] != "なし": continue
        url = r["テキストファイルURL"]
        if not url.endswith(".zip"): continue
        keep.append(r)
    return keep

RUBY = re.compile(r"《[^》]*》")
NOTE = re.compile(r"［＃[^］]*］")
BAR = re.compile(r"｜")
HEADER_SEP = re.compile(r"^-{10,}\s*$")

def clean(text: str) -> str:
    lines = text.splitlines()
    # ヘッダ: 最初の 2 本の "-----" 行に挟まれた注記を落とす
    seps = [i for i, l in enumerate(lines[:60]) if HEADER_SEP.match(l)]
    if len(seps) >= 2:
        lines = lines[seps[1] + 1:]
    # フッタ: "底本：" 以降を落とす
    for i, l in enumerate(lines):
        if l.startswith("底本：") or l.startswith("底本:"):
            lines = lines[:i]
            break
    text = "\n".join(lines)
    text = RUBY.sub("", text)
    text = NOTE.sub("", text)
    text = BAR.sub("", text)
    text = re.sub(r"[　 \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"

def fetch(row):
    wid = row["作品ID"]
    out = WORKS / f"{wid}.txt"
    if out.exists():
        return wid, out.stat().st_size, "cached"
    url = row["テキストファイルURL"]
    gh = url.replace("https://www.aozora.gr.jp/", "https://raw.githubusercontent.com/aozorabunko/aozorabunko/master/")
    data = None
    for u in (gh, url):
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "slm-ja-1m corpus builder"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            break
        except Exception as e:
            err = e
            time.sleep(1)
    if data is None:
        return wid, 0, f"failed: {err}"
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        raw = zf.read(name)
        text = raw.decode("cp932", errors="ignore")
    except Exception as e:
        return wid, 0, f"bad zip: {e}"
    cleaned = clean(text)
    out.write_text(cleaned, encoding="utf-8")
    return wid, len(cleaned), "ok"

def main():
    rows = load_index()
    by_author = {}
    for r in rows:
        by_author.setdefault(f"{r['姓']} {r['名']}", []).append(r)
    chosen = []
    for a in AUTHORS:
        chosen.extend(by_author.get(a, []))
    random.Random(0).shuffle(chosen)
    print(f"candidates: {len(chosen)} works from {len(AUTHORS)} authors")
    total = 0
    done = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        for wid, size, status in ex.map(fetch, chosen):
            if status.startswith("ok") or status == "cached":
                total += size
                done.append(wid)
            else:
                print("  skip", wid, status)
            if total >= TARGET:
                break
    print(f"works: {len(done)}, chars: {total}")
    with open(ROOT / "corpus.txt", "w", encoding="utf-8") as f:
        for wid in done:
            f.write((WORKS / f"{wid}.txt").read_text(encoding="utf-8"))
            f.write("\n\n")

if __name__ == "__main__":
    main()
