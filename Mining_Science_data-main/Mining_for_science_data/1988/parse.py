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


# TEXT_PATH = "output/1988 Measurement of the Compressibility and Sound Velocity of Helium up to 1 GPa, Kortbeek et al., Int. J. Thermophys.txt"

# with open(TEXT_PATH, 'r', encoding='utf-8') as file:
#     text = file.read()
# print("Successfully read the text file")

# # Get base output path without extension
# base_output_path = os.path.splitext(TEXT_PATH)[0]

# # Save extracted data
# save_extracted_data(text, base_output_path)


## test

import re
import json

def is_table_caption_line(line):
    """
    Checks if `line` starts with 'Table' followed by a roman numeral or digit,
    optionally a period (.) or colon (:), and then possibly more text.
    Examples of valid lines:
      - "Table I."
      - "Table I: Some caption"
      - "Table 2."
      - "Table III Some caption"
    """
    pattern = r'(?i)^(Table)\s+([IVXLCDM]+|\d+)([\.:])?\s*(.*)$'
    return re.match(pattern, line.strip()) is not None

def build_printed_table(table_data):
    """
    Given 'table_data' as a list of lists (rows, each row is a list of cell strings),
    build a nicely aligned pipe-delimited string.
    """
    if not table_data:
        return ""

    # Determine number of columns
    num_cols = max(len(row) for row in table_data)
    # Compute max width for each column
    col_widths = [0] * num_cols
    for row in table_data:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Construct each line
    lines_out = []
    for row in table_data:
        # Left-justify each cell to the width of that column
        row_str = "| " + " | ".join(
            row[i].ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i])
            for i in range(num_cols)
        ) + " |"
        lines_out.append(row_str)

    return "\n".join(lines_out)

def extract_tables_and_captions(text):
    """
    Scans through the input 'text', locates Markdown/ASCII-style tables,
    and extracts captions by checking ONLY the line ABOVE each table.
    
    Logic:
      - If the line above starts with "Table <roman-or-digit>...", that is our caption.
      - Otherwise, caption = None.
    
    Returns a list of dictionaries with the extracted information:
       [
         {
           "table_index": 1,
           "start_line": <int>,
           "end_line": <int>,
           "caption": <string or None>,
           "printed_table": <str>
         },
         ...
       ]
    """
    lines = text.split('\n')

    all_tables = []        # Will store raw cells for each table
    table_blocks = []      # Will store (start_line, end_line) for each table

    current_table = []
    in_table_block = False
    start_idx = None

    # 1) Identify each table and its start–end lines in the text
    for idx, line in enumerate(lines):
        line_stripped = line.strip()
        # Check if it's a table-like line
        if (
            line_stripped.startswith('|')
            and line_stripped.count('|') >= 2
            and not re.match(r'^\|[\s-]+\|$', line_stripped)
        ):
            if not in_table_block:
                in_table_block = True
                start_idx = idx  # mark beginning of the table block

            row_parts = line.split('|')
            # Remove empty leading/trailing cells if the line starts & ends with '|'
            # and strip whitespace around each cell
            row_cells = [c.strip() for c in row_parts if c.strip() != '']
            current_table.append(row_cells)
        else:
            if in_table_block:
                # We just reached the end of a table block
                in_table_block = False
                end_idx = idx - 1  # last line of the table was the previous line
                all_tables.append(current_table)
                table_blocks.append((start_idx, end_idx))
                current_table = []

    # Edge case: if the text ended while still parsing a table
    if in_table_block and current_table:
        end_idx = len(lines) - 1
        all_tables.append(current_table)
        table_blocks.append((start_idx, end_idx))

    # 2) For each extracted table, figure out its caption line
    final_tables = []
    for i, table_data in enumerate(all_tables, start=1):
        start_line, end_line = table_blocks[i-1]

        # Only check the line above
        line_above_idx = start_line - 1
        caption_line = lines[line_above_idx].strip() if line_above_idx >= 0 else ""

        # If above line looks like a caption, use it; otherwise None
        caption = caption_line if is_table_caption_line(caption_line) else None

        # Build a pretty-printed version of the table
        printed_table = build_printed_table(table_data)

        final_tables.append({
            "table_index": i,
            "start_line": start_line,
            "end_line": end_line,
            "caption": caption,
            "printed_table": printed_table
        })

    return final_tables


sample_text = r'''
Our data do not support the conclusion of Mills et al. that along an isotherm the product of the molar volume V and the sound velocity w is almost constant; e.g., for 298.15 K and 100 MPa the product wV=49.6 mg'kmol-ls -1, whereas at 1 GPa the value has decreased to 29.4 m 4' kmo1-1 9 s -1. But the deviations from Rao's rule, expressed by *Vw 1/3 =*  constant, as well as the free volume model, *wV 1/3* =constant, are even larger. Kimura et al. [21] proposed a modification of the sound velocity equation for an ideal gas, w= *(7opV/M) 1/2,* in the form w= *(7opV/M+ b(Vo/V)2) ~/2,* where 7o = 1.67, M is the molecular weight, and Vo is the molar volume at atmospheric pressure. While deviations between the sound velocity calculated by the ideal-gas taw and the experimental values range from 18 to 35 %, the largest deviation of the modified equation is about 1.2% and is found at the lowest pressure (100 MPa). The deviations are positive and decrease to about 0.3 % with increasing pressure up to 1 GPa

| p (MPa) | He | Ne | Ar | N 2 | CH 4 |
| --- | --- | --- | --- | --- | --- |
| 100 | 6.64 | 4.35 | 18.3 | 19.6 | 24.3 |
| 500 | 0.321 | 4.22 | 5.32 | 5.89 | 6.25 |
| 1000 | 0.165 | 2.47 | -- | 3.11 | 3.66 |

Table III. The Relative Change of the Sound Velocity per Degree K, e x 10 4, Along Several Isobars

For the results from 20 to 68 MPa, we initially used the data of Michels and Wouters [201 as reference data for densities below 20 MPa. It turned out that our results were not consistent with these data. For example, the density that we obtain from a total expansion with an initial pressure of 20 MPa (and a pressure after expansion of about 8.5 MPa) is

| p | p | w | ZT 105 | Xs 105 |  |
| --- | --- | --- | --- | --- | --- |
| (MPa) | (kmol.m -s) | (m.s 1) | (MPa 1) | (MPa-X) |  |
| 100 | 28.104 | 1393.35 | 719.58 | 457.90 | 1.571 |
| 150 | 37.134 | 1542.77 | 435.79 | 282.67 | 1.542 |
| 200 | 44.497 | 1674.74 | 303.24 | 200.19 | 1.515 |
| 250 | 50.755 | 1793.45 | 228.12 | 153.04 | 1.491 |
| 300 | 56.187 | 1901.57 | 180.72 | 122.97 | 1.470 |
| 350 | 60.991 | 2001.02 | 148.47 | 102.30 | 1.451 |
| 400 | 65.293 | 2093.24 | 125.28 | 87.33 | 1.435 |
| 450 | 69.184 | 2179.31 | 107.91 | 76.03 | 1.419 |
| 500 | 72.745 | 2260.10 | 94.47 | 67.24 | 1.405 |
| 550 | 76.048 | 2336.29 | 83.79 | 60.19 | 1.392 |
| 600 | 79.129 | 2408.44 | 75.13 | 54.43 | 1.380 |
| 650 | 81.989 | 2477.01 | 67.97 | 49.66 | 1.369 |
| 700 | 84.699 | 2542.37 | 61.97 | 45.64 | 1.358 |
| 750 | 87.268 | 2604.85 | 56.88 | 42.19 | 1.348 |
| 800 | 89.667 | 2664.73 | 52.51 | 39.24 | 1.338 |
| 850 | 91.988 | 2722.24 | 48.72 | 36.65 | 1.329 |
| 900 | 94.199 | 2777.58 | 45.40 | 34.38 | 1.321 |
| 950 | 96.264 | 2830.94 | 42.48 | 32.38 | 1.312 |
| 1000 | 98.264 | 2882.46 | 39.89 | 30.60 | 1.304 |

Table I. Thermodynamic Properties of Helium at 298.15 K

Table II. Velocity of Sound (in m.s -1) in Helium at Various Pressures (in MPa)

|  |  |  |  |  | T(K) |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p (MPa) | 298.15 | 273.15 | 248.12 | 223.17 |  | 198.15 173.15 | 148.15 | 123.10 | 98.10 |
| 100 | 1393.35 | 1368.80 | 1344.92 | 1320.50 | 1297.84 |  | 1282.32 1256.36 | 1240.14 | 1229.98 |
| 150 | 1542.77 | 1523.13 |  | 1504.81 1485.66 |  | 1470.01 1457.02 | 1443.67 | 1435.79 | 1433.14 |
| 200 | 1674.74 | 1659.57 | 1644.28 | 1629.59 |  | 1618.51 1608.28 1601.40 |  |  | 1598.11 1599.69 |
| 250 | 1793.45 | 1781.09 | 1768.60 | 1757.39 | 1749.50 | 1741.66 |  | 1738.48 1737.98 1742.00 |  |
| 300 | 1901.57 | 1891.16 | 1881.15 | 1872.56 |  | 1867.01 1861.11 1860.24 |  | 1861.54 | 1866.97 |
| 350 | 2001.02 | 1992.10 |  | 1984.21 1977.57 | 1973.80 | 1969.42 | 1970.12 | 1972.62 | 1978.84 |
| 400 | 2093.24 | 2085.52 | 2079.42 | 2074.23 | 2071.85 | 2068.67 | 2070.48 | 2073.81 | 2080.44 |
| 450 | 2179.31 | 2t72.62 | 2168.03 | 2163.93 | 2162.64 | 2160.39 | 2163.04 | 2166.93 | 2173~74 |
| 500 | 2260.10 | 2254.29 | 2250.97 | 2247.70 | 2247.28 | 2245.75 | 2249.05 | 2253.33 | 2260.18 |
| 550 | 2336.29 | 2331.26 | 2329.00 | 2326.38 | 2326.64 | 2325.69 | 2329.49 | 2334.03 | 2340.84 |
| 600 | 2408.44 | 2404.10 | 2402.73 | 2400.63 | 2401.43 | 2400.92 | 2405.13 | 2409.83 | 2416.55 |
| 650 | 2477.01 | 2473.28 | 2472.64 | 2470.99 | 2472.20 | 2472.05 | 2476.58 | 2481.36 | 2487.98 |
| 700 | 2542.37 | 2539.18 | 2539.16 | 2537.90 | 2539.42 | 2539.56 | 2544.34 | 2549.15 | 2555.66 |
| 750 | 2604.85 | 2602.13 | 2602.62 | 2601.72 | 2603.48 | 2603.84 | 2608.81 | 2613.61 | 2620.01 |
| 800 | 2664.73 | 2662.41 | 2663.33 | 2662.78 | 2664.69 | 2665.24 | 2670.35 | 2675.10 | 2681.41 |
| 850 | 2722.24 | 2720.26 | 2721.53 | 2721.33 | 2723.33 | 2724.04 | 2729.24 | 2733.91 |  |
| 900 | 2777.58 | 2775.89 | 2777.44 | 2777.61 | 2779.63 | 2780.49 | 2785.73 | 2790.30 |  |
| 950 | 2830.94 | 2829.47 | 2831.25 | 2831.81 | 2833.81 | 2834.79 | 2840.04 |  |  |
| t000 | 2882.46 | 2881.16 | 2883.13 | 2884.11 | 2886.03 | 2887.13 |  |  |  |

systematically lower than the density reported by Michels and Wouters at 20 MPa. The discrepancy is larger than can be explained by the errors in our procedure. It proves, in our opinion, that Michels and Wouters' results are not consistent within the stated accuracy. However, our data were found to be consistent with those of Briggs. A comparison of the results of this author with those of Michels and Wouters shows that the relative deviations of Michels and Wouters' data increase almost linearly from 0 at p=0 to 0.18% at 20MPa. Michels and Wouters used a glass tube piezometer and probably there was a loss of helium due to diffusion through the glass wall. Therefore, in this work the results of Briggs have been taken for the low-pressure densities.

'''

# Extract the tables + captions
tables_info = extract_tables_and_captions(sample_text)

# Print to verify
for info in tables_info:
    print(f"=== Table {info['table_index']} ===")
    print("Start line:", info["start_line"])
    print("End line:  ", info["end_line"])
    print("Caption:   ", info["caption"])
    print(info["printed_table"])
    print()

# If you want, save to JSON
with open("extracted_tables.json", "w", encoding="utf-8") as f:
    json.dump(tables_info, f, indent=4)

print("JSON output written to 'extracted_tables.json'")

