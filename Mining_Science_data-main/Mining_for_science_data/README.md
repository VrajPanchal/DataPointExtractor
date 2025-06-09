# Mining_Science_data

1. Step I: Parse the PDF to extract tables, figures
   (1) install PDF parsor library Marker: https://github.com/VikParuchuri/marker

   pip install marker-pdf
   python parse.py

--The outputs:
   - a .txt file of the paper in outlined layer format
   - .dat table files including all the tables in the paper pdf
   - .jpeg figures including all the figures in the paper pdf
   - .json file including data of captions for tables, captions for figures, math equations

Tests: 
Now we have three tests with paper pdfs about physics. The results are in the folders of "1984", "1988", "2008"
-1. 1984 Shock Compression of Liquid Helium to 56 GPa (560 kbar), Nellis et al., Phys. Rev. Lett
-2. 1988 Measurement of the Compressibility and Sound Velocity of Helium up to 1 GPa, Kortbeek et al., Int. J. Thermophys
-3. 2008 Fluid helium at conditions of giant planetary interiors, Stixrude and Jeanloz, Proc. Natl. Acad. Sci

Optional: 
  (2) install PDF parsor library MinerU: https://github.com/opendatalab/MinerU


2. Step II: analyze the tables and figures information to genrate accurate table data for each table and figure for further understanding

   
