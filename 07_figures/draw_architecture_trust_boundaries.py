from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "fig1_architecture_trust_boundaries.png"
W, H = 2600, 1120
im = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(im)
font_dir = Path(r"C:\Windows\Fonts")
regular = ImageFont.truetype(str(font_dir / "arial.ttf"), 41)
bold = ImageFont.truetype(str(font_dir / "arialbd.ttf"), 45)
small = ImageFont.truetype(str(font_dir / "arial.ttf"), 36)
note = ImageFont.truetype(str(font_dir / "arial.ttf"), 34)

navy = "#203047"
blue = "#EAF3FC"
green = "#EAF6EF"
amber = "#FFF4D6"
red = "#B64040"
gray = "#526174"


def centered(box, text, font, fill=navy, gap=7):
    x1, y1, x2, y2 = box
    lines = text.split("\n")
    sizes = [d.textbbox((0, 0), line, font=font) for line in lines]
    total = sum(b[3] - b[1] for b in sizes) + gap * (len(lines) - 1)
    y = (y1 + y2 - total) / 2
    for line, b in zip(lines, sizes):
        h = b[3] - b[1]
        d.text(((x1 + x2 - (b[2] - b[0])) / 2, y - b[1]), line, font=font, fill=fill)
        y += h + gap


def box(rect, title, body, fill, outline, body_bounds=None):
    d.rounded_rectangle(rect, radius=22, fill=fill, outline=outline, width=5)
    x1, y1, x2, y2 = rect
    d.text((x1 + 18, y1 + 13), title, font=bold, fill=outline)
    bounds = body_bounds or (x1 + 18, y1 + 68, x2 - 18, y2 - 10)
    centered(bounds, body, small, navy, 10)


def arrow(x1, y1, x2, y2, color=gray, dashed=False):
    if dashed:
        length = math.hypot(x2 - x1, y2 - y1)
        steps = max(1, int(length / 24))
        for i in range(0, steps, 2):
            a = i / steps
            b = min((i + 1) / steps, 1.0)
            d.line((x1 + a * (x2 - x1), y1 + a * (y2 - y1),
                    x1 + b * (x2 - x1), y1 + b * (y2 - y1)), fill=color, width=7)
    else:
        d.line((x1, y1, x2, y2), fill=color, width=8)
    angle = math.atan2(y2 - y1, x2 - x1)
    size = 25
    points = [(x2, y2),
              (x2 - size * math.cos(angle - 0.5), y2 - size * math.sin(angle - 0.5)),
              (x2 - size * math.cos(angle + 0.5), y2 - size * math.sin(angle + 0.5))]
    d.polygon(points, fill=color)


centered((50, 15, W - 50, 82),
         "Edge ASR Command Path and Trust Boundaries", bold, navy)

# Trust domains: QCS8275 (including its HTP), local IPC, Arduino Bridge, and STM32.
domains = [
    ((420, 130, 1320, 760), "QCS8275", "#F5F8FC", "#3A6EA5"),
    ((1340, 130, 1690, 760), "Local IPC", "#F9FAFC", "#69798C"),
    ((1710, 130, 2075, 760), "Arduino RPC", "#F4F8F7", "#38805D"),
    ((2095, 130, 2540, 760), "STM32", green, "#38805D"),
]
for rect, title, fill, outline in domains:
    d.rounded_rectangle(rect, radius=25, fill=fill, outline=outline, width=6)
    d.text((rect[0] + 22, rect[1] + 16), title, font=bold, fill=outline)

box((55, 330, 380, 545), "Audio", "16-kHz PCM\nVIVOS WAV", amber, "#A57A18")
box((470, 260, 920, 385), "Feature extraction", "80-bin log-Mel", blue, "#3A6EA5")
box((470, 415, 860, 575), "CPU inference", "Whisper-Tiny\nW8A16 encoder /\nINT8 decoder", blue, "#3A6EA5")
box((1010, 415, 1275, 575), "Hexagon HTP", "QNN partial\nexecution", blue, "#3A6EA5")
box((470, 620, 920, 750), "ASR output", "Text / command ID", amber, "#A57A18")

box((1370, 280, 1660, 440), "Router", "Method routing", "#F2F4F7", "#69798C")
box((1370, 520, 1660, 680), "UNIX socket", "Local IPC", "#F2F4F7", "#69798C")
box((1740, 300, 2045, 650), "RPC bridge", "Arduino Bridge\nMPU to STM32\nDownstream path\nnot traced", "#EEF6F1", "#38805D")
box((2125, 300, 2510, 650), "Action endpoint", "STM32 command path\nPhysical actuation\nnot tested", green, "#38805D")

# Data flow: the local router is measured; downstream RPC/actuation is shown as documented but untested.
arrow(380, 435, 455, 325)
arrow(695, 385, 695, 415)
arrow(860, 495, 1000, 495, "#3A6EA5")
centered((860, 447, 1005, 487), "FastRPC", note, "#3A6EA5")
arrow(695, 575, 695, 620)
arrow(920, 685, 1360, 600, "#A57A18")
arrow(1515, 520, 1515, 450)
arrow(1660, 360, 1730, 475, "#69798C", dashed=True)
arrow(2045, 475, 2115, 475, "#38805D", dashed=True)

# Red dashed lines mark trust-domain boundaries.
for x in (405, 1328, 1698, 2083):
    for y in range(215, 755, 35):
        d.line((x, y, x, y + 18), fill=red, width=4)

centered((35, 775, 2565, 845),
         "Trust domains: audio ingress | QCS8275 CPU/HTP | local IPC/router | Arduino RPC bridge | STM32",
         small, red)
d.rounded_rectangle((55, 860, 2545, 1050), radius=18, fill="#F4F6F8", outline="#AAB4C0", width=3)
centered((75, 875, 2525, 1035),
         "FastRPC is inside QCS8275 (CPU to HTP); Arduino RPC bridges the MPU and STM32.\nMeasured boundary: local router only. Downstream bridge behavior and actuation were not tested.",
         note, navy, 12)

im.save(OUT, dpi=(220, 220), optimize=True)
print(OUT)
