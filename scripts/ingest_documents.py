"""
Simple ingestion script to add local documents into the vector store.

Usage examples (PowerShell):

# Ingest all .md and .txt files under Agri/docs and save KB to knowledge_base
python .\scripts\ingest_documents.py -i "Agri/docs" -o "Agri/knowledge_base" --ext md,txt

# Ingest rows from a CSV (title/content columns)
python .\scripts\ingest_documents.py -c "some_rows.csv" --csv-title-col title --csv-content-col content -o "Agri/knowledge_base"

By default the script will process documents and save them to the output directory.
"""

import os
import sys
import argparse
import glob
import json
import pandas as pd

# Ensure project package import works
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
agri_pkg_dir = os.path.join(project_root, 'Agri')
if agri_pkg_dir not in sys.path:
    sys.path.insert(0, agri_pkg_dir)

def ingest_from_folder(folder: str, exts: list):
    added = 0
    documents = []
    for ext in exts:
        pattern = os.path.join(folder, f"**/*.{ext}")
        for path in glob.glob(pattern, recursive=True):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
                    text = fh.read().strip()
                title = os.path.relpath(path)
                if text:
                    documents.append({'title': title, 'content': text})
                    added += 1
            except Exception as e:
                print(f"Failed to ingest {path}: {e}")
    return added, documents

def ingest_from_csv(csv_path: str, title_col: str, content_col: str):
    added = 0
    documents = []
    df = pd.read_csv(csv_path)
    if title_col not in df.columns or content_col not in df.columns:
        raise ValueError(f"CSV does not contain specified columns: {title_col}, {content_col}")
    for _, row in df.iterrows():
        title = str(row[title_col])
        content = str(row[content_col])
        if content and len(content.strip()) > 20:
            documents.append({'title': title, 'content': content})
            added += 1
    return added, documents

def ingest_from_jsonl(jsonl_path: str, title_field: str = 'title', content_field: str = 'content'):
    added = 0
    documents = []
    with open(jsonl_path, 'r', encoding='utf-8') as fh:
        for line in fh:
            try:
                obj = json.loads(line)
                title = obj.get(title_field, '') or ''
                content = obj.get(content_field, '') or ''
                if content and len(content.strip()) > 20:
                    documents.append({'title': title, 'content': content})
                    added += 1
            except Exception as e:
                print(f"Skipping bad JSON line: {e}")
    return added, documents

def save_documents(documents, output_path):
    """Save documents to JSON file"""
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, 'documents.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(documents)} documents to {output_file}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input-dir', help='Directory to ingest files from (recursive)', default=None)
    p.add_argument('--ext', help='Comma-separated list of file extensions to ingest (md,txt,csv,json,...)', default='md,txt')
    p.add_argument('-c', '--csv', help='Path to CSV file to ingest rows from', default=None)
    p.add_argument('--csv-title-col', help='CSV column to use for title', default='title')
    p.add_argument('--csv-content-col', help='CSV column to use for content', default='content')
    p.add_argument('--jsonl', help='Path to JSONL file to ingest', default=None)
    p.add_argument('--jsonl-title-field', help='JSONL field for title', default='title')
    p.add_argument('--jsonl-content-field', help='JSONL field for content', default='content')
    p.add_argument('-o', '--output-path', help='Directory to save the documents to', default='Agri/knowledge_base')
    args = p.parse_args()

    all_documents = []
    total_added = 0

    if args.input_dir:
        exts = [e.strip().lower() for e in args.ext.split(',') if e.strip()]
        print(f"Ingesting from folder {args.input_dir} extensions: {exts}")
        added, documents = ingest_from_folder(args.input_dir, exts)
        all_documents.extend(documents)
        print(f"Added {added} documents from folder")
        total_added += added

    if args.csv:
        print(f"Ingesting from CSV {args.csv}")
        added, documents = ingest_from_csv(args.csv, args.csv_title_col, args.csv_content_col)
        all_documents.extend(documents)
        print(f"Added {added} rows from CSV")
        total_added += added

    if args.jsonl:
        print(f"Ingesting from JSONL {args.jsonl}")
        added, documents = ingest_from_jsonl(args.jsonl, args.jsonl_title_field, args.jsonl_content_field)
        all_documents.extend(documents)
        print(f"Added {added} items from JSONL")
        total_added += added

    if total_added == 0:
        print("No documents were added. Nothing to save.")
    else:
        # Save documents
        print(f"Saving documents to {args.output_path} ...")
        try:
            save_documents(all_documents, args.output_path)
            print("Save completed.")
        except Exception as e:
            print(f"Failed to save documents: {e}")

    print(f"Done. Total documents added: {total_added}")

if __name__ == '__main__':
    main()