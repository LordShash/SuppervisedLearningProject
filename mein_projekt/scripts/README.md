# CSV File Splitting Script

This script is used to split large CSV files into smaller chunks to avoid exceeding the configured file size limit in the IDE (2.56 MB).

## Background

The original CSV file `Daten_UTF8_Clean_encoded.csv` was 3.27 MB, which exceeded the configured limit of 2.56 MB, causing code insight features to be unavailable. This script splits the file into two smaller parts, each below the limit.

## Usage

To split the CSV file, run the following command from the project root directory:

```
python scripts/split_csv.py
```

This will create two new files in the data directory:
- `Daten_UTF8_Clean_encoded_part1.csv`
- `Daten_UTF8_Clean_encoded_part2.csv`

## Data Loading

The `data_loader.py` file has been modified to support both the original file and the split files. It will automatically detect and load the split files if they exist, or fall back to the original file if they don't.

## File Sizes

- Original file: 3.27 MB
- Split files: ~1.61 MB and ~1.66 MB

Both split files are well below the configured limit of 2.56 MB, which resolves the issue with code insight features.