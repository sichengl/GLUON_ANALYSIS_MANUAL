# FIT_PLAN.md -> FIT_PLAN.pdf (A4).  Step 1 (anaconda python, has "markdown"): md -> html.  Step 2 (/usr/bin/python3 -E, has fitz): html -> pdf
#   python md_to_pdf.py html FIT_PLAN.md FIT_PLAN.html
#   /usr/bin/python3 -E md_to_pdf.py pdf FIT_PLAN.html FIT_PLAN.pdf
import sys
CSS = """
body { font-family: sans-serif; font-size: 9.5pt; line-height: 1.35; color: #1f1e1d; }
h1 { font-size: 15pt; margin: 0 0 6pt 0; }
h2 { font-size: 12pt; margin: 12pt 0 4pt 0; border-bottom: 0.5pt solid #b4b2a9; }
p { margin: 3pt 0; }
ul, ol { margin: 2pt 0 2pt 0; }
li { margin: 1pt 0; }
table { border-collapse: collapse; margin: 4pt 0; }
th, td { border: 0.5pt solid #b4b2a9; padding: 1.5pt 5pt; font-size: 8.5pt; }
th { font-weight: bold; }
code { font-family: monospace; font-size: 8.5pt; }
"""
mode, src, dst = sys.argv[1], sys.argv[2], sys.argv[3]
if mode == "html":
    import markdown
    body = markdown.markdown(open(src).read(), extensions=["tables"])
    open(dst, "w").write(body)
else:
    import fitz
    html = open(src).read()
    story = fitz.Story(html=html, user_css=CSS)
    page = fitz.paper_rect("a4")
    where = page + (50, 50, -50, -50)
    writer = fitz.DocumentWriter(dst)
    more = True
    while more:
        dev = writer.begin_page(page)
        more, _ = story.place(where)
        story.draw(dev)
        writer.end_page()
    writer.close()
    doc = fitz.open(dst)
    bad = sum(1 for p in doc for b in p.get_text("blocks") if b[0] < 40 or b[2] > p.rect.x1 - 40 or b[1] < 40 or b[3] > p.rect.y1 - 40)
    print(dst, "pages", len(doc), "| text blocks outside the margins:", bad)
