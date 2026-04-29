from __future__ import annotations

import csv
from pathlib import Path

from certificate_reader.domain import CSV_COLUMNS, ExtractionResult


def write_results_csv(results: list[ExtractionResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_csv_row())
