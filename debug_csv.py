import csv

with open("contact_logs.csv", newline='', encoding="utf-8") as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader, 1):
        print(f"Line {i} ({len(row)} fields): {row}")
