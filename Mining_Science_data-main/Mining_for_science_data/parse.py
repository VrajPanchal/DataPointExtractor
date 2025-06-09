from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

import os
import re
from typing import List, Dict
import json  # Add this import at the top


def save_to_text(file_path: str, content: str) -> None:
    """
    Save extracted content to a text file with the same name as the input PDF.
    
    Args:
        file_path: Path to the input PDF file
        content: Text content to save
    """
    # Change file extension from .pdf to .txt
    DIR_path = "output"
    base_name = os.path.splitext(file_path)[0]
    output_path = f"{DIR_path}/{base_name}.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Saved extracted content to: {output_path}")

##=====extract tables, equations, figures==============================================================================================  

def extract_equations(text: str) -> List[str]:
    """
    Extract mathematical equations from scientific text.
    Equations are typically enclosed in $$ or $ symbols.
    """
    # Pattern for equations: $$...$$ or $...$ or \begin{equation}...\end{equation}
    equation_pattern = r'\$\$.*?\$\$|\$.*?\$|\\begin{equation}.*?\\end{equation}'
    equations = re.findall(equation_pattern, text, re.DOTALL)
    
    # Clean up equations by removing extra whitespace
    return [eq.strip() for eq in equations]

def extract_tables(text):
    """
    Scans through the input 'text', locates Markdown/ASCII-style tables 
    (lines starting with '|' and containing multiple columns), parses them, 
    and prints them back in a neatly aligned, pipe-delimited format. 

    Returns a list of tables; each table is a list of rows, 
    and each row is a list of string cells.
    """
    lines = text.split('\n')
    
    # This list will hold lists of parsed tables;
    # each table is a list-of-lists (rows, then cells).
    all_tables = []
    
    current_table = []
    in_table_block = False

    for line in lines:
        # A quick test to see if the line is "table-like":
        #   - starts with '|'
        #   - has at least 2 '|' characters
        #   - is not just a line of dashes (the separator row in Markdown).
        table_candidate = line.strip()
        if (table_candidate.startswith('|') 
            and table_candidate.count('|') >= 2 
            and not re.match(r'^\|[\s-]+\|$', table_candidate)
        ):
            # We are in a table row
            in_table_block = True

            # Parse the row by splitting on '|'
            # The first split and the last split may be empty if the line starts & ends with '|'
            row_parts = line.split('|')
            
            # Remove empty leading/trailing cells if the line starts & ends with '|'
            # Also strip each cell to remove extra whitespace
            row_cells = [cell.strip() for cell in row_parts if cell.strip() != '']
            
            current_table.append(row_cells)
        else:
            # Not a table-like line
            if in_table_block:
                # Just ended a table block, so store the current table
                if current_table:
                    all_tables.append(current_table)
                current_table = []
                in_table_block = False
            # else: just keep going, ignoring non-table lines

    # If the text ended and we were still in a table block,
    # close out the last table.
    if current_table:
        all_tables.append(current_table)

    # Now 'all_tables' is a list of tables in raw cell format.
    # Let's print them out with column widths that fit the content.
    printed_tables = []
    for table in all_tables:
        # First, we must figure out how many columns the table has 
        # and the maximum width of each column.
        # table is a list of rows, each row is a list of cell strings
        # Some rows might have fewer or more cells than others (though typically they do not).
        
        num_cols = max(len(r) for r in table)
        
        # Compute max width per column
        col_widths = [0]*num_cols
        for row in table:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
        
        # Build a string representation of each row
        lines_out = []
        for row in table:
            # For each cell, left-justify according to col_widths
            row_str = "| " + " | ".join(
                cell.ljust(col_widths[i]) for i, cell in enumerate(row)
            ) + " |"
            lines_out.append(row_str)
        
        # Join all lines for this table into one string
        table_str = "\n".join(lines_out)
        printed_tables.append(table_str)
    
    # Print or return the tables in a nicely formatted manner.
    # Here, let's just print them with a blank line separating each.
    for idx, t in enumerate(printed_tables, 1):
        print(f"--- Table {idx} ---")
        print(t)
        print()  # blank line after each table
    
    # Return the list-of-lists-of-lists if you need them programmatically
    return all_tables, printed_tables

def save_table(table_str, output_filename):
    """
    Saves the given table string to the specified output filename.
    """
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

def extract_figure_captions(text: str) -> List[Dict]:
    """
    Extract figure captions from scientific text using image reference pattern.
    Returns a list of dictionaries with figure type, number, and caption.
    """
    # First find all image references
    image_pattern = r'!\[\]\((_page_\d+_Figure_\d+\.jpeg)\)'
    image_matches = re.findall(image_pattern, text)
    
    extracted_figures = []
    for img_ref in image_matches:
        # Find the text after the image reference
        caption_pattern = rf'!\[\]\({re.escape(img_ref)}\)\s*\n(.*?)(?=\n\n|!\[\]|$)'
        caption_match = re.search(caption_pattern, text, re.DOTALL)
        
        if caption_match:
            caption_text = caption_match.group(1).strip()
            # Extract figure number from caption
            number_match = re.search(r'(Fig\.?|Figure|fig\.?|figure)\s+([\d\.]+)', caption_text)
            if number_match:
                # Clean the number by removing any trailing punctuation
                clean_number = re.sub(r'[^\d]', '', number_match.group(2))
                extracted_figures.append({
                    'image_reference': img_ref,
                    # 'type': number_match.group(1),
                    'number': clean_number,  # Now only contains digits
                    'caption': caption_text
                })
    
    return extracted_figures


def save_extracted_data(text: str, base_output_path: str) -> None:
    """
    Save extracted equations, tables, and figure captions to output files.
    
    Args:
        text: The text content to extract from
        base_output_path: Base path for output files (without extension)
    """
    # Extract data
    equations = extract_equations(text)
    all_tables, printed_tables = extract_tables(text)
    figures = extract_figure_captions(text)
    
    # Save equations and figures to JSON
    json_data = {
        'equations': equations,
        'figures': figures
    }
    json_path = f"{base_output_path}_metadata.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=4)
    print(f"Saved equations and figure captions to: {json_path}")
    
    # Save tables to .dat files
    DIR_path = "output/tables"
    os.makedirs(DIR_path, exist_ok=True)  # Ensure directory exists
    
    for idx, t in enumerate(printed_tables, 1):
        table_filename = f"{DIR_path}/table_{idx}.dat"
        save_table(t, table_filename)
        print(f"Saved table {idx} to: {table_filename}")
##=======USING marker OCR ways==============================================================================================

# FILE_PATH = "1988 Measurement of the Compressibility and Sound Velocity of Helium up to 1 GPa, Kortbeek et al., Int. J. Thermophys.pdf"
# FILE_PATH = "1984 Shock Compression of Liquid Helium to 56 GPa (560 kbar), Nellis et al., Phys. Rev. Lett.pdf"
# FILE_PATH = "2008 Fluid helium at conditions of giant planetary interiors, Stixrude and Jeanloz, Proc. Natl. Acad. Sci.pdf"

# converter = PdfConverter(
#     artifact_dict=create_model_dict(),
# )
# rendered = converter(FILE_PATH)
# text, _, images = text_from_rendered(rendered)
# print("Text:" ,text)
# print("Images:" ,images)

# ## Save texts and figures
# save_to_text(FILE_PATH, text)
# for filename, img in images.items():
#     img.save(filename)

# print(f"Saved extracted figures")

##=====extract tables, equations, figures


TEXT_PATH = "output/2008 Fluid helium at conditions of giant planetary interiors, Stixrude and Jeanloz, Proc. Natl. Acad. Sci.txt"

with open(TEXT_PATH, 'r', encoding='utf-8') as file:
    text = file.read()
print("Successfully read the text file")

# Get base output path without extension
base_output_path = os.path.splitext(TEXT_PATH)[0]

# Save extracted data
save_extracted_data(text, base_output_path)



## test


# import re
# import json

# def is_table_caption_line(line):
#     """
#     Checks if `line` starts with 'Table' followed by a roman numeral or digit,
#     optionally a period (.) or colon (:), and then possibly more text.
#     Examples of valid lines:
#       - "Table I."
#       - "Table I: Some caption"
#       - "Table 2."
#       - "Table III Some caption"
#     """
#     pattern = r'(?i)^(Table)\s+([IVXLCDM]+|\d+)([\.:])?\s*(.*)$'
#     return re.match(pattern, line.strip()) is not None

# def build_printed_table(table_data):
#     """
#     Given 'table_data' as a list of lists (rows, each row is a list of cell strings),
#     build a nicely aligned pipe-delimited string.
#     """
#     if not table_data:
#         return ""

#     # Determine number of columns
#     num_cols = max(len(row) for row in table_data)
    
#     # Compute max width for each column
#     col_widths = [0] * num_cols
#     for row in table_data:
#         for i, cell in enumerate(row):
#             col_widths[i] = max(col_widths[i], len(cell))

#     # Construct each line in a pipe-delimited manner
#     lines_out = []
#     for row in table_data:
#         row_str = "| " + " | ".join(
#             row[i].ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i])
#             for i in range(num_cols)
#         ) + " |"
#         lines_out.append(row_str)

#     return "\n".join(lines_out)

# def extract_tables_and_captions(text):
#     """
#     Scans through the input 'text', locates Markdown/ASCII-style tables,
#     and extracts captions by looking ABOVE each table. Specifically:
#       - Skip any blank lines immediately above.
#       - Then if we find a line matching "Table <Roman-or-Digit>..." we use it.
#       - Otherwise, caption = None.

#     Returns a list of dictionaries:
#        [
#          {
#            "table_index": <int>,
#            "start_line":  <int>,
#            "end_line":    <int>,
#            "caption":     <str or None>,
#            "printed_table": <str>
#          },
#          ...
#        ]
#     """
#     lines = text.split('\n')

#     all_tables = []        # Will store raw cells for each table
#     table_blocks = []      # Will store (start_line, end_line) for each table

#     current_table = []
#     in_table_block = False
#     start_idx = None

#     # 1) Identify each table and its start–end lines in the text
#     for idx, line in enumerate(lines):
#         line_stripped = line.strip()
#         if (
#             line_stripped.startswith('|')
#             and line_stripped.count('|') >= 2
#             and not re.match(r'^\|[\s-]+\|$', line_stripped)
#         ):
#             if not in_table_block:
#                 in_table_block = True
#                 start_idx = idx  # mark beginning of the table block

#             row_parts = line.split('|')
#             # Remove empty leading/trailing cells if the line starts & ends with '|'
#             # and strip whitespace around each cell
#             row_cells = [c.strip() for c in row_parts if c.strip() != '']
#             current_table.append(row_cells)
#         else:
#             if in_table_block:
#                 # We just reached the end of a table block
#                 in_table_block = False
#                 end_idx = idx - 1  # last line of the table was the previous line
#                 all_tables.append(current_table)
#                 table_blocks.append((start_idx, end_idx))
#                 current_table = []

#     # Edge case: if the text ended while still parsing a table
#     if in_table_block and current_table:
#         end_idx = len(lines) - 1
#         all_tables.append(current_table)
#         table_blocks.append((start_idx, end_idx))

#     # 2) For each extracted table, determine its caption
#     final_tables = []
#     for i, table_data in enumerate(all_tables, start=1):
#         start_line, end_line = table_blocks[i - 1]

#         # We'll step upwards from the line just above the table
#         # until we find a non-blank line (or run out of lines).
#         caption = None
#         check_idx = start_line - 1
#         while check_idx >= 0:
#             candidate_line = lines[check_idx].strip()
#             if candidate_line:  # not blank
#                 # If it's a caption line, store it; otherwise None.
#                 if is_table_caption_line(candidate_line):
#                     caption = candidate_line
#                 break
#             check_idx -= 1

#         # Build a pretty-printed version of the table
#         printed_table = build_printed_table(table_data)

#         final_tables.append({
#             "table_index": i,
#             "start_line": start_line,
#             "end_line": end_line,
#             "caption": caption,
#             "printed_table": printed_table
#         })

#     return final_tables


# if __name__ == "__main__":
#     sample_text = r'''
# Table II. Velocity of Sound (in m.s -1) in Helium at Various Pressures (in MPa)

# |  |  |  |  |  | T(K) |  |  |  |  |
# | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
# | p (MPa) | 298.15 | 273.15 | 248.12 | 223.17 |  | 198.15 173.15 | 148.15 | 123.10 | 98.10 |
# | 100 | 1393.35 | 1368.80 | 1344.92 | 1320.50 | 1297.84 |  | 1282.32 1256.36 | 1240.14 | 1229.98 |
# ...
# | t000 | 2882.46 | 2881.16 | 2883.13 | 2884.11 | 2886.03 | 2887.13 |  |  |  |

# systematically lower than the density reported by Michels and Wouters at 20 MPa. 
# ...
# Therefore, in this work the results of Briggs have been taken for the low-pressure densities.
# '''

#     # Extract tables
#     tables_info = extract_tables_and_captions(sample_text)

#     # Print them out
#     for info in tables_info:
#         print(f"=== Table {info['table_index']} ===")
#         print("Start line:", info["start_line"])
#         print("End line:  ", info["end_line"])
#         print("Caption:   ", repr(info["caption"]))
#         print(info["printed_table"])
#         print()

#     # Optionally save to JSON
#     with open("extracted_tables.json", "w", encoding="utf-8") as f:
#         json.dump(tables_info, f, indent=2)
    
#     print("JSON output written to 'extracted_tables.json'")
