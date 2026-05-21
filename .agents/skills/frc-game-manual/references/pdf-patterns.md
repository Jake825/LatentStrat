# Extracting Data from Manual PDFs

The FRC Game Manual is highly visual and uses multi-column layouts, diagrams, and tables. Standard text extraction can lose context. Use PDF extraction for exact wording and visual rendering for geometry.

## Assets

Use these bundled PDFs for 2026 REBUILT work:

- `assets/manuals/2026GameManual-TU22.pdf`
- `assets/manuals/REBUILT_TeamUpdate-Combined.pdf`

Resolve paths relative to the `frc-game-manual` skill directory.

## Extract Rule Text with pdfplumber

```python
import re
from pathlib import Path

import pdfplumber


def extract_rule_text(pdf_path: str | Path, rule_id: str) -> str | None:
    rule_pattern = re.compile(rf"^{re.escape(rule_id)}\b\s*(.*)", re.IGNORECASE)

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if rule_id.upper() not in text.upper():
                continue

            lines = text.splitlines()
            for index, line in enumerate(lines):
                if rule_pattern.match(line.strip()):
                    context = "\n".join(lines[index : index + 15])
                    return f"Page {page.page_number}:\n{context}"

    return None
```

## Search Manual and Team Updates Together

```python
from pathlib import Path

import pdfplumber


def search_pdfs(pdf_paths: list[str | Path], query: str) -> list[dict[str, object]]:
    hits = []
    needle = query.lower()

    for pdf_path in map(Path, pdf_paths):
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if needle in text.lower():
                    hits.append(
                        {
                            "pdf": pdf_path.name,
                            "page": page.page_number,
                            "snippet": text[:2000],
                        }
                    )

    return hits
```

## Extract Tables

```python
import pandas as pd
import pdfplumber


def extract_tables_from_page(pdf_path: str, page_number: int) -> list[pd.DataFrame]:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        tables = []

        for table in page.extract_tables():
            if not table or len(table) < 2:
                continue
            tables.append(pd.DataFrame(table[1:], columns=table[0]))

        return tables
```

## Render Diagrams

Render pages before answering spatial questions about field dimensions, AprilTags, robot size limits, bumper cross sections, or protected zones.

```python
import matplotlib.pyplot as plt
import pdfplumber


def render_page(pdf_path: str, page_number: int, resolution: int = 150) -> None:
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number - 1]
        image = page.to_image(resolution=resolution)

    plt.figure(figsize=(10, 14))
    plt.imshow(image.original)
    plt.axis("off")
    plt.show()
```

## Practical Rules

- Extract exact text for rule citations.
- Render diagrams when geometry matters.
- Check the Team Updates PDF for the same section before finalizing.
- If local PDF tooling is missing, state that limitation and cite source links rather than guessing.
