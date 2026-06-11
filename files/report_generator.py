from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import io


def generate_pdf(detections: list[dict]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("WasteVision AI — Detection Report", styles["Title"]))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    story.append(Spacer(1, 20))

    headers = ["#", "Class", "Confidence", "Recyclable", "Hazard", "Timestamp"]
    rows = [headers]
    for i, d in enumerate(detections, 1):
        rows.append([
            str(i),
            d.get("waste_class", ""),
            f"{d.get('confidence', 0):.2%}",
            "Yes" if d.get("recyclable") else "No",
            d.get("hazard_level", ""),
            d.get("timestamp", "")[:19],
        ])

    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a2a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f8f4")]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    story.append(t)

    total = len(detections)
    recyclable = sum(1 for d in detections if d.get("recyclable"))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Total Detections: {total}", styles["Normal"]))
    story.append(Paragraph(f"Recyclable: {recyclable} ({recyclable/max(total,1):.0%})", styles["Normal"]))

    doc.build(story)
    return buf.getvalue()
