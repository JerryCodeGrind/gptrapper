import csv

with open("instruments_db.csv", newline='') as f:
    reader = csv.reader(f)
    first_row = next(reader)
    print("Number of columns:", len(first_row))