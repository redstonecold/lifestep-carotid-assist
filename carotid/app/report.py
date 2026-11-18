"""분석 보고서(HTML → PDF) — 계획서 14절 `보고서` (필수 2, 선택 8).

- 상태: `AI 분석 초안`(기본) 또는 `의료진 검토 완료`(승인 후, 검토자·일시·의견 표시)
- 내용: 영상 정보 · 요약값 · 내경 그래프 · 근거 프레임(신뢰 박동의 최대·최소 내경, 검토 구간 대표 장면) · 박동 표 · 검토 목록
- 근거 프레임에는 앱과 같은 측정 그래픽(경계 · 대표 내경 캘리퍼)을 그린다
- PDF는 이 컴퓨터의 Chrome(헤드리스)으로 HTML을 인쇄해 만든다. Chrome이 없으면 HTML만 만든다
"""
import html
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np

CHROME = ["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
          "/Applications/Chromium.app/Contents/MacOS/Chromium",
          "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"]
RKO = {"no_vessel": "혈관 없음", "not_long": "종단면 아님", "low_conf": "확신도 낮음", "diam_jump": "내경 급변",
       "motion": "탐촉자 이동", "shape": "모양 급변"}
BKO = {"beat_valid": "제외 프레임 많음", "beat_len": "주기 이상", "beat_amp": "변화량 이상", "edge": "영상 끝"}


def _fx(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def evidence_image(gray: np.ndarray, contour: list, frame: dict, out: Path, unit: str):
    """근거 프레임: 경계(하늘색) + 대표 내경 캘리퍼(노랑) + 내경 수치."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    s = img.shape[1] / 730
    good = not frame["reasons"]
    if len(contour) > 2:
        cv2.polylines(img, [np.array(contour, np.int32)], True, (240, 200, 120) if good else (60, 150, 235),
                      max(1, round(1.4 * s)), cv2.LINE_AA)
    g = frame.get("geo")
    if g:
        P = lambda t, q: (int(round(g["c"][0] + t * g["u"][0] + q * g["v"][0])), int(round(g["c"][1] + t * g["u"][1] + q * g["v"][1])))
        t, q0, q1 = g["med"]
        a, b = P(t, q0 - .5), P(t, q1 + .5)
        yel = (76, 201, 242) if good else (76, 160, 232)
        cv2.line(img, a, b, yel, max(1, round(1.5 * s)), cv2.LINE_AA)
        for x, y in (a, b):
            r = round(5 * s)
            cv2.line(img, (x - r, y), (x + r, y), yel, max(1, round(1.5 * s)), cv2.LINE_AA)
            cv2.line(img, (x, y - r), (x, y + r), yel, max(1, round(1.5 * s)), cv2.LINE_AA)
        up = a if a[1] < b[1] else b
        cv2.putText(img, f"D {frame['diam_mm']:.2f} {unit.replace('*', '')}", (up[0] + round(7 * s), up[1] - round(5 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, yel, max(1, round(1.2 * s)), cv2.LINE_AA)
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def chart_svg(res: dict, unit: str, W=760, H=200) -> str:
    fr = res["frames"]
    T = fr[-1]["t"] or 1
    ok = [f["diam_mm"] for f in fr if not f["reasons"] and f["diam_mm"] is not None]
    allv = [f["diam_mm"] for f in fr if f["diam_mm"] is not None]
    src = ok if len(ok) > 5 else allv or [0, 10]
    lo, hi = min(src), max(src)
    pad = max(.3, (hi - lo) * .3); lo -= pad; hi += pad
    l, r, t, b = 40, 10, 8, 22
    X = lambda v: l + (W - l - r) * v / T
    Y = lambda d: max(t, min(H - b - 8, t + (H - t - b - 8) * (1 - (d - lo) / (hi - lo))))
    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Pretendard" font-size="10" fill="#8691a1">']
    step = _nice((hi - lo) / 4)
    v = np.ceil(lo / step) * step
    while v <= hi:
        s.append(f'<line x1="{l}" x2="{W - r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e8ebef"/><text x="{l - 6}" y="{Y(v) + 3:.1f}" text-anchor="end">{v:.1f}</text>')
        v += step
    ts = _nice(T / 10); v = 0.0
    while v <= T + 1e-9:
        s.append(f'<text x="{X(v):.1f}" y="{H - 6}" text-anchor="middle">{v:g}</text>'); v += ts
    s.append(f'<line x1="{l}" x2="{W - r}" y1="{H - b}" y2="{H - b}" stroke="#b7c0cc"/><text x="{W - r}" y="{H - 6}" text-anchor="end">s</text><text x="4" y="{t + 8}">{unit}</text>')
    for seg in res["no_vessel"]:
        s.append(f'<rect x="{X(seg["t0"]):.1f}" y="{H - b - 5}" width="{max(1.5, X(seg["t1"]) - X(seg["t0"])):.1f}" height="5" fill="#8d96a3"/>')
    for seg in res["review"]:
        if seg["kind"] == "구간":
            s.append(f'<rect x="{X(seg["t0"]):.1f}" y="{H - b - 5}" width="{max(1.5, X(seg["t1"]) - X(seg["t0"])):.1f}" height="5" fill="#c27a12"/>')
    for bt in res["beats"]:
        s.append(f'<line x1="{X(bt["t0"]):.1f}" x2="{X(bt["t0"]):.1f}" y1="{t}" y2="{H - b}" stroke="{"#1d7a4f" if bt["reliable"] else "#a86400"}" opacity=".5"{"" if bt["reliable"] else " stroke-dasharray=\"3 3\""}/>')
    for f in fr:
        if f["diam_mm"] is None:
            continue
        if not f["reasons"]:
            s.append(f'<circle cx="{X(f["t"]):.1f}" cy="{Y(f["diam_mm"]):.1f}" r="1.5" fill="#1d4f8c"/>')
        else:
            x, y = X(f["t"]), Y(f["diam_mm"])
            s.append(f'<path d="M{x - 1.5:.1f} {y - 1.5:.1f}l3 3m0 -3l-3 3" stroke="#c0392b" stroke-width=".9"/>')
    s.append("</svg>")
    return "".join(s)


def _nice(x):
    p = 10 ** np.floor(np.log10(x)); n = x / p
    return float((1 if n < 1.5 else 2 if n < 3.5 else 5 if n < 7.5 else 10) * p)


def build(job: dict, res: dict, files: list, out_dir: Path, base_url: str) -> Path:
    """보고서 HTML(+근거 이미지)을 out_dir에 만든다. 반환: HTML 경로."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m, s = res["meta"], res["summary"]
    unit = "mm" if m["mm_calibrated"] else "mm*"
    ok = s["status"] == "자동측정"
    fr = res["frames"]
    # 근거 프레임: 신뢰 박동 중 변화량이 중앙값에 가까운 박동의 최대·최소 + 검토 구간 대표 장면 최대 2곳
    ev = []
    good = [b for b in res["beats"] if b["reliable"]]
    if good:
        med = np.median([b["delta"] for b in good])
        b = min(good, key=lambda x: abs(x["delta"] - med))
        seg = [i for i, f in enumerate(fr) if b["t0"] <= f["t"] <= b["t1"] and not f["reasons"] and f["diam_mm"] is not None]
        if seg:
            imax = max(seg, key=lambda i: fr[i]["diam_mm"]); imin = min(seg, key=lambda i: fr[i]["diam_mm"])
            ev += [(imax, f"대표 박동 {res['beats'].index(b) + 1} · 최대 내경(수축기)"), (imin, f"대표 박동 {res['beats'].index(b) + 1} · 최소 내경(이완기)")]
    for r in [r for r in res["review"] if r["kind"] == "구간"][:2]:
        ev.append((r["frame"], "검토 필요 · " + ", ".join(r["reasons_ko"])))
    figs = []
    for k, (i, cap) in enumerate(ev):
        p = out_dir / f"ev{k}.jpg"
        evidence_image(cv2.imread(files[i], cv2.IMREAD_GRAYSCALE), res["contours"][i], fr[i], p, unit)
        d = fr[i]["diam_mm"]
        figs.append(f'<figure><img src="{p.name}"><figcaption><b>{html.escape(cap)}</b><span>{fr[i]["t"]:.2f} s · 프레임 {i + 1} · 내경 {_fx(d)} {unit}</span></figcaption></figure>')

    appr = job.get("approval")
    edited = sum(1 for f in fr if f.get("edited"))
    stamp = (f'<div class="stamp done">의료진 검토 완료</div>' if appr else '<div class="stamp">AI 분석 초안 · 의료진 검토 전</div>')
    rows = lambda pairs: "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in pairs)
    meta_rows = rows([("분석 번호", job["id"]), ("영상", html.escape(job["name"])), ("형식 · 장비", f'{m["format"]} · {html.escape(m["device"])}'),
                      ("길이 · 프레임", f'{m["duration_s"]:.1f} s · {m["n_frames"]} 프레임'),
                      ("mm 보정", f'{m["mm_per_px"]:.4f} mm/px · {html.escape(m["mm_source"])}'),
                      ("모델", html.escape(m.get("model", ""))), ("분석 일시", job["created"]),
                      ("사람이 고친 프레임", f"{edited}장" if edited else "없음")])
    cell = lambda k, v, u="": f'<div><span>{k}</span><b>{v}<small>{u}</small></b></div>'
    summary = (f'<div class="verdict {"ok" if ok else "no"}"><span>판정</span><b>{s["status"]}</b></div>'
               + cell("최소 내경 · 이완기", _fx(s.get("dmin_mm")) if ok else "—", unit)
               + cell("최대 내경 · 수축기", _fx(s.get("dmax_mm")) if ok else "—", unit)
               + cell("변화량", _fx(s.get("delta_mm")) if ok else "—", unit)
               + cell("심박수", _fx(s.get("hr_bpm"), 0) if ok else "—", "bpm")
               + cell("신뢰 박동", s["n_reliable"], f"/ {s['n_beats']}"))
    beat_rows = "".join(
        f'<tr><td class="n">{i + 1}</td><td class="n">{_fx(b["t0"])}</td><td class="n">{_fx(b["dmin"])}</td><td class="n">{_fx(b["dmax"])}</td>'
        f'<td class="n">{_fx(b["delta"])}</td><td class="n">{_fx(b["dur_s"])}</td>'
        f'<td>{"신뢰" if b["reliable"] else "검토 · " + ", ".join(BKO.get(r, r) for r in b["reasons"])}</td></tr>'
        for i, b in enumerate(res["beats"])) or '<tr><td colspan="7">박동을 찾지 못함</td></tr>'
    rev_rows = "".join(f'<tr><td class="n">{_fx(r["t0"])} – {_fx(r["t1"])}</td><td>{r["kind"]}</td><td>{", ".join(r["reasons_ko"])}</td></tr>'
                       for r in res["review"]) or '<tr><td colspan="3">없음</td></tr>'
    notes = []
    if not m["mm_calibrated"]:
        notes.append(f"mm* — {html.escape(m['mm_source'])}. 수치는 참고값입니다.")
    if not ok:
        notes.append("믿을 수 있는 박동이 2개 미만이라 요약값을 내지 않았습니다.")
    sign = (f'<div class="sign"><div><span>검토자</span><b>{html.escape(appr["reviewer"])}</b></div>'
            f'<div><span>검토 일시</span><b>{appr["time"]}</b></div>'
            f'<div class="wide"><span>의견</span><p>{html.escape(appr.get("comment") or "—")}</p></div></div>'
            if appr else '<div class="sign empty"><div><span>검토자</span><b>&nbsp;</b></div><div><span>검토 일시</span><b>&nbsp;</b></div><div class="wide"><span>의견</span><p>&nbsp;</p></div></div>')

    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>경동맥 내경 측정 보고서 {job['id']}</title>
<style>
@font-face {{ font-family: Pretendard; src: url("{base_url}/static/fonts/PretendardVariable.woff2") format("woff2-variations"); font-weight: 45 920; }}
@page {{ size: A4; margin: 14mm 14mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; font: 10pt/1.5 Pretendard, sans-serif; color: #111927; font-feature-settings: "tnum"; }}
header {{ display: flex; align-items: flex-end; justify-content: space-between; border-bottom: 2px solid #0b1f3a; padding-bottom: 8px; }}
header h1 {{ margin: 0; font-size: 17pt; font-weight: 700; color: #0b1f3a; letter-spacing: -.02em; }}
header p {{ margin: 2px 0 0; color: #465366; font-size: 9pt; }}
.stamp {{ border: 1.5px solid #a86400; color: #a86400; padding: 4px 10px; font-weight: 700; font-size: 10pt; }}
.stamp.done {{ border-color: #143a6b; color: #fff; background: #143a6b; }}
h2 {{ font-size: 11pt; margin: 16px 0 6px; color: #0b1f3a; }}
table.meta {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 9pt; }}
table.meta th {{ text-align: left; color: #8691a1; font-weight: 500; width: 22%; padding: 2px 0; }}
table.meta td {{ padding: 2px 0; }}
.summary {{ display: grid; grid-template-columns: 1.2fr repeat(5, 1fr); border: 1px solid #d5dae1; margin-top: 12px; }}
.summary > div {{ padding: 7px 10px; border-left: 1px solid #e8ebef; }} .summary > div:first-child {{ border-left: 0; }}
.summary span {{ display: block; font-size: 8pt; color: #8691a1; }}
.summary b {{ font-size: 16pt; font-weight: 600; }} .summary small {{ font-size: 8pt; color: #8691a1; margin-left: 2px; font-weight: 500; }}
.summary .verdict.ok {{ background: #143a6b; }} .summary .verdict.no {{ background: #a86400; }}
.summary .verdict span {{ color: rgba(255,255,255,.75); }} .summary .verdict b {{ color: #fff; font-size: 13pt; }}
.note {{ margin: 6px 0 0; padding-left: 8px; border-left: 3px solid #a86400; color: #7a4a00; font-size: 9pt; }}
.figs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
figure {{ margin: 0; break-inside: avoid; }} figure img {{ width: 100%; display: block; }}
figcaption {{ font-size: 8.5pt; padding-top: 3px; }} figcaption b {{ display: block; }} figcaption span {{ color: #8691a1; }}
table.t {{ width: 100%; border-collapse: collapse; font-size: 9pt; }}
table.t th {{ text-align: left; color: #8691a1; font-weight: 600; border-bottom: 1px solid #b7c0cc; padding: 3px 6px; }}
table.t td {{ border-bottom: 1px solid #e8ebef; padding: 3px 6px; }} .n {{ text-align: right; }}
.key {{ font-size: 8pt; color: #465366; }} .key i {{ display: inline-block; width: 9px; height: 4px; margin: 0 3px 1px 10px; }}
.sign {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px 20px; border: 1px solid #d5dae1; padding: 10px 12px; break-inside: avoid; }}
.sign span {{ display: block; font-size: 8pt; color: #8691a1; }} .sign b {{ display: block; min-height: 18px; border-bottom: 1px solid #b7c0cc; }}
.sign .wide {{ grid-column: 1 / -1; }} .sign p {{ margin: 0; min-height: 30px; border-bottom: 1px solid #b7c0cc; }}
footer {{ margin-top: 14px; font-size: 8pt; color: #8691a1; }}
.pb {{ break-before: page; }}
</style></head><body>
<header><div><h1>경동맥 내경 측정 보고서</h1><p>Carotid lumen diameter · 연구용 소프트웨어 · 진단을 대체하지 않음</p></div>{stamp}</header>
<table class="meta">{meta_rows}</table>
<div class="summary">{summary}</div>
{"".join(f'<p class="note">{n}</p>' for n in notes)}
<h2>프레임별 내경</h2>{chart_svg(res, unit)}
<div class="key"><i style="background:#1d4f8c;height:6px;width:6px;border-radius:50%"></i>측정 프레임<i style="background:#c0392b"></i>제외 프레임<i style="background:#c27a12"></i>검토 필요 구간<i style="background:#8d96a3"></i>혈관 없음·횡단면<i style="background:#1d7a4f;width:2px;height:9px"></i>박동 시작</div>
<h2>근거 장면</h2><div class="figs">{"".join(figs) or "<p>표시할 장면 없음</p>"}</div>
<h2 class="pb">박동별 내경 ({unit})</h2>
<table class="t"><thead><tr><th class="n">#</th><th class="n">시작 s</th><th class="n">최소</th><th class="n">최대</th><th class="n">변화량</th><th class="n">주기 s</th><th>판정</th></tr></thead><tbody>{beat_rows}</tbody></table>
<h2>검토 필요 ({len(res["review"])}곳)</h2>
<table class="t"><thead><tr><th class="n">시간 s</th><th>종류</th><th>사유</th></tr></thead><tbody>{rev_rows}</tbody></table>
<h2>판독 확인</h2>{sign}
<footer>측정 방법: 내강 분할 → 혈관 축 가운데 50% 구간의 수직 폭 중앙값. 신뢰도 판정: 프레임(혈관 없음·횡단면·확신도) · 시간(내경·위치·모양 급변) · 박동(제외 비율·주기·변화량). 믿을 수 있는 박동만 요약에 사용.</footer>
</body></html>"""
    p = out_dir / "report.html"
    p.write_text(doc, encoding="utf-8")
    return p


def to_pdf(html_url: str, pdf: Path) -> bool:
    chrome = next((c for c in CHROME if Path(c).exists()), None) or shutil.which("google-chrome") or shutil.which("chromium")
    if not chrome:
        return False
    r = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=5000",
                        f"--print-to-pdf={pdf}", html_url], capture_output=True, timeout=90)
    return pdf.exists() and pdf.stat().st_size > 0
