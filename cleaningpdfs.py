from PyPDF2 import PdfReader
from fpdf import FPDF
import os

# Define the paths to the original PDFs
pdf_1_original_path = r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\20241101 Angus Gerro Audio\AUDIO_Transcript_1.pdf"
pdf_2_original_path = r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\20241101 Angus Gerro Audio\AUDIO_Transcript_2.pdf"

# Helper function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Extract text from the original PDFs
content_1_original = extract_text_from_pdf(pdf_1_original_path)
content_2_original = extract_text_from_pdf(pdf_2_original_path)

# Add improved structure and headings to the original content (manually structuring based on analysis)
content_1 = f"""
Audio Transcript - Conversation 1: Structured Version

1. Introduction to the Discussion
{content_1_original[:300]}

2. Flow State in Team Games
{content_1_original[300:700]}

3. Balancing Team Dynamics
{content_1_original[700:1200]}

4. Game Design Considerations
{content_1_original[1200:]}
"""

content_2 = f"""
Audio Transcript - Conversation 2: Structured Version

1. Introduction
{content_2_original[:300]}

2. Adapting Games for Player Experience
{content_2_original[300:700]}

3. Role of Technology in Coaching
{content_2_original[700:1100]}

4. Structuring Effective Training Sessions
{content_2_original[1100:]}
"""

# Function to create a structured PDF from given content
def create_pdf(title, content, output_path, font_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.add_font("FreeSerif", '', font_path, uni=True)  # Add the Unicode-compatible font
    pdf.set_font("FreeSerif", '', 16)
    pdf.cell(200, 10, title, ln=True, align='C')
    
    pdf.set_font("FreeSerif", '', 12)
    pdf.ln(10)
    
    # Add headings and paragraphs
    lines = content.split('\n')
    for line in lines:
        if line.strip() == "":
            pdf.ln(5)
        else:
            pdf.multi_cell(0, 10, line)

    pdf.output(output_path)

# Create structured PDFs with original content and added headings
output_folder_path = r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\20241101 Angus Gerro Audio"
pdf_1_structured_path = os.path.join(output_folder_path, "AUDIO_Transcript_1_Structured.pdf")
pdf_2_structured_path = os.path.join(output_folder_path, "AUDIO_Transcript_2_Structured.pdf")

# Path to the FreeSerif font
font_path = r"C:\Users\13459\Dropbox\Businesses\Rugby Tribe\20241101 Angus Gerro Audio\FreeSerif.ttf"

# Create the PDFs using the specified font
create_pdf("Audio Transcript - Conversation 1: Structured Version", content_1, pdf_1_structured_path, font_path)
create_pdf("Audio Transcript - Conversation 2: Structured Version", content_2, pdf_2_structured_path, font_path)

print(f"PDFs created successfully:\n1. {pdf_1_structured_path}\n2. {pdf_2_structured_path}")
