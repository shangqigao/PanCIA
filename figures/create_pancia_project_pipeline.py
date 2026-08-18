"""Create a presentation-ready PNG overview of the complete PanCIA workflow."""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 3600, 1800
BG, WHITE, INK, MUTED, LINE = "#F7F9FC", "#FFFFFF", "#14213D", "#5D6B82", "#D8E0EB"
BLUE, BLUE_PALE = "#2477C9", "#E9F4FF"
PINK, PINK_PALE = "#B33D7A", "#FCECF4"
TEAL, TEAL_PALE = "#087F8C", "#E5F6F5"
GOLD, GOLD_PALE = "#D48A16", "#FFF5DE"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_B = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size)


def rounded(draw, xy, fill, outline=LINE, width=3, radius=28):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def text(draw, xy, value, size, color=INK, bold=False, anchor="la", spacing=10):
    draw.multiline_text(xy, value, font=font(size, bold), fill=color,
                        anchor=anchor, spacing=spacing)


def card(draw, xy, title, body, fill, edge, accent, title_size=27, body_size=21):
    x1, y1, x2, y2 = xy
    rounded(draw, xy, fill, edge, 3, 26)
    draw.rounded_rectangle((x1, y1, x1 + 11, y2), radius=5, fill=accent)
    text(draw, (x1 + 30, y1 + 35), title, title_size, accent, True)
    text(draw, (x1 + 30, y1 + 86), body, body_size, MUTED, False, spacing=9)


def arrow(draw, start, end, color="#93A1B5", width=7, head=20):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2 - head, y2), fill=color, width=width)
    draw.polygon([(x2, y2), (x2 - head, y2 - head * .62),
                  (x2 - head, y2 + head * .62)], fill=color)


def down_arrow(draw, start, end, color="#93A1B5", width=6, head=18):
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2 - head), fill=color, width=width)
    draw.polygon([(x2, y2), (x2 - head * .62, y2 - head),
                  (x2 + head * .62, y2 - head)], fill=color)


def phase(draw, x, n, label, color):
    draw.ellipse((x, 285, x + 54, 339), fill=color)
    text(draw, (x + 27, 312), str(n), 23, WHITE, True, "mm")
    text(draw, (x + 75, 312), label.upper(), 23, color, True, "lm")


def graph_icon(draw, cx, cy, color):
    pts = [(cx-35, cy-20), (cx+33, cy-24), (cx-44, cy+34), (cx+3, cy+12), (cx+45, cy+39)]
    for i, j in [(0,1),(0,2),(0,3),(1,3),(1,4),(2,3),(3,4)]:
        draw.line((*pts[i], *pts[j]), fill=color, width=4)
    for x, y in pts:
        draw.ellipse((x-9, y-9, x+9, y+9), fill=WHITE, outline=color, width=4)


def main():
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)

    text(d, (150, 105), "PanCIA", 64, INK, True)
    text(d, (500, 116), "Pan-Cancer Image Analysis Pipeline", 50, INK, True)
    text(d, (153, 205),
         "Radiology + digital pathology  →  graph-based patient representation  →  clinically relevant outcomes",
         28, MUTED)
    d.line((150, 260, 3450, 260), fill=LINE, width=3)

    phase(d, 190, 1, "Cohort", BLUE)
    phase(d, 850, 2, "Modality processing", PINK)
    phase(d, 1930, 3, "Representation learning", TEAL)
    phase(d, 2860, 4, "Outcome modelling", GOLD)

    columns = [(120, 370, 700, 1480), (790, 370, 1810, 1480),
               (1870, 370, 2720, 1480), (2780, 370, 3480, 1480)]
    for c in columns:
        rounded(d, c, WHITE, "#E3E9F1", 3, 35)

    # Cohort
    card(d, (190, 455, 630, 670), "Radiology", "CT / MRI volumes\nDICOM → NIfTI",
         BLUE_PALE, "#A9D2F5", BLUE, 31, 24)
    card(d, (190, 735, 630, 950), "Pathology", "Whole-slide images\nQuality screening",
         PINK_PALE, "#E9B4CE", PINK, 31, 24)
    down_arrow(d, (410, 675), (410, 1015), "#A8B5C6", 5, 18)
    card(d, (190, 1025, 630, 1235), "Matched cohort", "Inclusion / exclusion\nSubject harmonization",
         "#F2F5F9", LINE, INK, 29, 23)
    arrow(d, (630, 560), (850, 560), BLUE)
    arrow(d, (630, 840), (850, 840), PINK)

    # Processing
    card(d, (850, 455, 1240, 675), "Tumour segmentation", "Automated masks\nTumour + margin",
         BLUE_PALE, "#A9D2F5", BLUE, 27, 22)
    arrow(d, (1240, 565), (1340, 565), BLUE)
    card(d, (1350, 455, 1750, 675), "Radiomics", "Slice / tumour\nMulti-scale features",
         BLUE_PALE, "#A9D2F5", BLUE, 29, 22)
    card(d, (850, 735, 1240, 955), "WSI tiling", "Tissue patches\nSpatial coordinates",
         PINK_PALE, "#E9B4CE", PINK, 29, 22)
    arrow(d, (1240, 845), (1340, 845), PINK)
    card(d, (1350, 735, 1750, 955), "Pathomics", "Patch embeddings\nMorphology + context",
         PINK_PALE, "#E9B4CE", PINK, 29, 22)
    card(d, (850, 1060, 1750, 1300), "Foundation feature encoders",
         "Radiology  ·  BiomedParse / LVMMed / FMCIB / PyRadiomics\nPathology  ·  UNI / CONCH / CHIEF",
         "#F5F1FF", "#CFC1EC", "#6952A3", 29, 22)
    arrow(d, (1750, 565), (1930, 565), BLUE)
    arrow(d, (1750, 845), (1930, 845), PINK)

    # Representation
    card(d, (1930, 455, 2250, 675), "Radiology graph", "Spatial nodes\nMulti-scale edges",
         BLUE_PALE, "#A9D2F5", BLUE, 25, 21)
    graph_icon(d, 2180, 585, BLUE)
    card(d, (1930, 735, 2250, 955), "Pathology graph", "Patch nodes\nTissue topology",
         PINK_PALE, "#E9B4CE", PINK, 25, 21)
    graph_icon(d, 2180, 865, PINK)
    arrow(d, (2250, 565), (2350, 690), BLUE, 6)
    arrow(d, (2250, 845), (2350, 720), PINK, 6)
    card(d, (2360, 600, 2660, 815), "Graph fusion", "SPARRA / GNN\nMultimodal",
         TEAL_PALE, "#9FD7D4", TEAL, 27, 22)
    down_arrow(d, (2510, 820), (2510, 950), TEAL, 7)
    card(d, (1950, 970, 2640, 1245), "Universal patient embedding",
         "Multimodal · multi-scale · multitask\nCompact patient-level representation",
         "#E8F3F4", "#7FC3C5", TEAL, 32, 23)
    for i, hh in enumerate([55, 90, 42, 105, 70, 98]):
        d.rounded_rectangle((2460+i*24, 1110-hh, 2473+i*24, 1110), radius=5, fill=TEAL)
    arrow(d, (2640, 1105), (2825, 1105), TEAL, 8, 24)

    # Outcomes
    card(d, (2860, 450, 3400, 690), "Survival prediction", "OS · DSS · DFI · PFI\nRisk stratification",
         "#FFF2E8", "#F2C49D", "#B85D18", 31, 23)
    card(d, (2860, 755, 3400, 995), "Phenotype prediction", "Immune / molecular subtype\nPrimary disease",
         "#F2EEFF", "#CBBCEC", "#6B4CB3", 31, 23)
    card(d, (2860, 1060, 3400, 1315), "Signature prediction",
         "Gene programs · HRD · immunity\nStemness · age regression",
         GOLD_PALE, "#E8C979", GOLD, 31, 23)
    arrow(d, (2825, 1105), (2860, 570), "#A7AFBC", 5, 18)
    arrow(d, (2825, 1105), (2860, 875), "#A7AFBC", 5, 18)
    arrow(d, (2825, 1105), (2860, 1185), "#A7AFBC", 5, 18)

    # Bottom statement
    rounded(d, (190, 1535, 3400, 1690), INK, INK, 0, 30)
    text(d, (270, 1612), "PAN-CANCER LEARNING", 23, "#9ED7E1", True, "lm")
    text(d, (820, 1612),
         "Reusable patient representations connect image morphology, spatial organization and clinical outcomes",
         28, WHITE, False, "lm")

    out = Path(__file__).with_name("pancia_project_pipeline.png")
    im.save(out, quality=95, optimize=True)
    print(out)


if __name__ == "__main__":
    main()
