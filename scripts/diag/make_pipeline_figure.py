"""Generate the README pipeline figure, results/figures/pipeline.svg (PNG: rsvg-convert -z 2).

Planner spec
  Stage 1 Data preparation : [WorldView-3 bands] -> [SpaceNet 2 products] -> [Co-registration] -> [Spatial split]
  Stage 2 Detection        : one LR image branches into two paths; the HR image feeds the HR detector
      LR image --------------------------------> Detector trained on LR --> (Baseline)
      LR image --> Super-resolution (8 models) --> Detector trained on HR --> (SR + detection)
      HR image ----------------------------------^ (same detector)      --> (Upper bound)
  Stage 3 Evaluation       : [Map-space merging] -> [Accuracy] -> [Gain vs. baseline]
Style guide
  white page, light-gray stage panels, white boxes with 1 px gray border, one muted-blue accent
  for the SR path, a title and at most two short lines per box, gray arrows, result names as pills.
"""
from html import escape
from pathlib import Path

W, H = 1200, 666
FONT = "Inter, Helvetica, Arial, sans-serif"
INK, SUB, MUTED = "#111827", "#4B5563", "#6B7280"
EDGE, PANEL = "#9CA3AF", "#F5F6F8"
ACC, ACC_FILL = "#2F5D9E", "#E6EDF7"

out = []
a = out.append


def text(x, y, s, size=13, weight=400, fill=SUB, anchor="start"):
    a(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{fill}" '
      f'text-anchor="{anchor}">{escape(s)}</text>')


def line(d, color=MUTED, width=1.2, head=True, marker="arr"):
    m = f' marker-end="url(#{marker})"' if head else ""
    a(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{m}/>')


STAGE = {1: ("#EEF6F5", "#2C7A73"), 2: ("#EFF3FA", "#2F5D9E"), 3: ("#F8F3EA", "#9A6B2F")}


def panel(y, h, num, name):
    fill, col = STAGE[num]
    a(f'<rect x="20" y="{y}" width="{W - 40}" height="{h}" rx="6" fill="{fill}"/>')
    a(f'<rect x="20" y="{y}" width="5" height="{h}" rx="2" fill="{col}"/>')
    text(38, y + 24, f"Stage {num}", 14, 700, col)
    text(106, y + 24, name, 14, 700, INK)


def box(x, y, w, h, title, lines=(), accent=False):
    stroke, sw = (ACC, 1.4) if accent else (EDGE, 1)
    a(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="#FFFFFF" '
      f'stroke="{stroke}" stroke-width="{sw}"/>')
    top = y + h / 2 - (len(lines) * 18) / 2 + 5
    text(x + w / 2, top, title, 15, 600, INK, "middle")
    for i, s in enumerate(lines):
        text(x + w / 2, top + 21 + 18 * i, s, 13, 400, SUB, "middle")


def pill(x, cy, w, label, accent=False):
    fill, stroke, col = (ACC_FILL, ACC, ACC) if accent else ("#FFFFFF", EDGE, INK)
    a(f'<rect x="{x}" y="{cy - 18}" width="{w}" height="36" rx="18" fill="{fill}" stroke="{stroke}"/>')
    text(x + w / 2, cy + 5, label, 14, 600, col, "middle")


a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  f'font-family="{FONT}">')
a('<defs>'
  f'<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
  f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{MUTED}"/></marker>'
  f'<marker id="arrA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
  f'orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="{ACC}"/></marker></defs>')
a(f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>')

# Stage 1: four boxes
panel(16, 140, 1, "Data preparation")
s1y, s1h, s1w, gap = 52, 88, 256, 32
x1 = [40 + i * (s1w + gap) for i in range(4)]
box(x1[0], s1y, s1w, s1h, "WorldView-3 bands",
    ["PAN 0.3 m (1 band)", "multispectral 1.2 m (8 bands)"])
box(x1[1], s1y, s1w, s1h, "SpaceNet 2 products",
    ["LR: multispectral RGB, 1.2 m", "HR: pan-sharpened RGB, 0.3 m"])
box(x1[2], s1y, s1w, s1h, "Co-registration",
    ["common UTM grid (GDAL)", "1 LR pixel = 4 × 4 HR pixels"])
box(x1[3], s1y, s1w, s1h, "Spatial split",
    ["576 m blocks", "train / val / test"])
for i in range(3):
    line(f"M{x1[i] + s1w + 3} {s1y + s1h / 2}H{x1[i + 1] - 5}")

# Stage 2: detection
py = 172
panel(py, 300, 2, "Detection")
yA, yB, yC = py + 70, py + 166, py + 258
IX, IW = 64, 180
SX, SW_ = 292, 420
DX, DW = 760, 222
TX, TW = 1022, 136
bh = 56

# inputs: one LR image feeding two paths, one HR image
box(IX, yA - bh / 2, IW, yB - yA + bh, "LR image")
box(IX, yC - bh / 2, IW, bh, "HR image")

# detectors
box(DX, yA - bh / 2, DW, bh, "Detector trained on LR")
box(DX, yB - bh / 2, DW, yC - yB + bh, "Detector trained on HR")

# super-resolution
sy, sh = yB - 66, 132
a(f'<rect x="{SX}" y="{sy}" width="{SW_}" height="{sh}" rx="5" fill="#FFFFFF" '
  f'stroke="{ACC}" stroke-width="1.4"/>')
text(SX + 16, sy + 26, "Super-resolution (8 models)", 15, 600, INK)
rows = [("Interpolation", "Bicubic, Lanczos", False),
        ("Regression", "SwinIR, Real-ESRNet", False),
        ("GAN", "Real-ESRGAN", False),
        ("Diffusion", "ResShift, OSEDiff", False),
        ("Fine-tuned GAN", "Real-ESRGAN, tuned on our image pairs", True)]
for i, (fam, models, hi) in enumerate(rows):
    yy = sy + 49 + 18 * i
    if hi:
        a(f'<rect x="{SX + 8}" y="{yy - 13}" width="{SW_ - 16}" height="19" rx="3" fill="{ACC_FILL}"/>')
    text(SX + 16, yy, fam, 13, 600, ACC if hi else INK)
    text(SX + 136, yy, models, 13, 600 if hi else 400, ACC if hi else SUB)

# result names
pill(TX, yA, TW, "Baseline")
pill(TX, yB, TW, "SR + detection", accent=True)
pill(TX, yC, TW, "Upper bound")

# arrows
line(f"M{IX + IW + 2} {yA}H{DX - 5}")
line(f"M{IX + IW + 2} {yB}H{SX - 5}", ACC, 1.4, marker="arrA")
line(f"M{SX + SW_ + 2} {yB}H{DX - 5}", ACC, 1.4, marker="arrA")
line(f"M{IX + IW + 2} {yC}H{DX - 5}")
line(f"M{DX + DW + 2} {yA}H{TX - 5}")
line(f"M{DX + DW + 2} {yB}H{TX - 5}", ACC, 1.4, marker="arrA")
line(f"M{DX + DW + 2} {yC}H{TX - 5}")

# feed from Stage 1 into the two inputs
BX = 44
line(f"M{x1[3] + s1w / 2} {s1y + s1h}V{py + 36}H{BX}V{yC}", EDGE, 1.2, head=False)
line(f"M{BX} {(yA + yB) / 2}H{IX - 5}")
line(f"M{BX} {yC}H{IX - 5}")

# Stage 3
ey0 = 488
panel(ey0, 162, 3, "Evaluation")
ey, eh = ey0 + 40, 96
xs, cw = (40, 424, 808), 352
RB, EX = 1170, xs[0] + cw - 60
line(f"M{TX + TW + 2} {yA}H{RB}V{ey0 - 8}H{EX}", EDGE, 1.2, head=False)
line(f"M{TX + TW + 2} {yB}H{RB}", EDGE, 1.2, head=False)
line(f"M{TX + TW + 2} {yC}H{RB}", EDGE, 1.2, head=False)
line(f"M{EX} {ey0 - 8}V{ey - 5}")

box(xs[0], ey, cw, eh, "Map-space merging", ["tile detections → building polygons"])
box(xs[1], ey, cw, eh, "Accuracy", ["AP50", "recall by building size"])
box(xs[2], ey, cw, eh, "Gain vs. baseline", ["AP50 difference with 95% CI", "(spatial block bootstrap)"])
line(f"M{xs[0] + cw + 3} {ey + eh / 2}H{xs[1] - 5}")
line(f"M{xs[1] + cw + 3} {ey + eh / 2}H{xs[2] - 5}")

a("</svg>")
OUT = Path(__file__).resolve().parents[2] / "results/figures/pipeline.svg"
OUT.write_text("\n".join(out), encoding="utf-8")
print(OUT)
