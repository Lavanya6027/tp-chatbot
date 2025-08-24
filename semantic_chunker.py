import re
from typing import List

class SemanticChunker:
    """
    Splits text into semantic chunks: headings + their content, 
    lists, and paragraphs.
    """

    def chunk(self, text: str) -> List[str]:
        lines = text.splitlines()
        chunks = []
        buffer = []

        heading_pattern = re.compile(r"^(\d+(\.\d+)*\s+|[A-Z][A-Z\s]+)$")  # e.g. "1. Scope", "OBJECTIVE"
        list_pattern = re.compile(r"^(\d+\.|[-*•])\s+")  # bullets / lists

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Headings → start new chunk
            if heading_pattern.match(line):
                if buffer:
                    chunks.append(" ".join(buffer).strip())
                    buffer = []
                buffer.append(line)
                continue

            # Bullets/lists → stay with current chunk
            if list_pattern.match(line):
                buffer.append(line)
                continue

            # Paragraph line → append
            buffer.append(line)

        if buffer:
            chunks.append(" ".join(buffer).strip())

        # Keep only meaningful chunks
        return [c for c in chunks if len(c.split()) > 5]
