import csv
import sys

csv.field_size_limit(sys.maxsize)

# Open the CSV file for reading
with open('./my_file.tsv', 'r') as file:
    reader = csv.reader(file, delimiter='\t')
    rows = list(reader)

# Open the CSV file for writing
with open('my_file.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
