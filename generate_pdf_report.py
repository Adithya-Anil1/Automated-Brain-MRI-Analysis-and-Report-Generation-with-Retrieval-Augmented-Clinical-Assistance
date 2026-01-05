#!/usr/bin/env python3
"""
Professional PDF Report Generator for MRI Brain Analysis

Generates publication-quality PDF reports from the template-driven
radiology report system.
"""

import os
import sys
import json
from datetime import datetime
from fpdf import FPDF


class ProfessionalMRIReport(FPDF):
    """
    Professional MRI Brain Report PDF Generator.
    
    Features:
    - Clean, clinical layout
    - Two-column demographics
    - Section-based organization
    - Electronic signature block
    - Disclaimer footer
    """
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)
    
    def header(self):
        """Professional header with title and institution line."""
        # Main Title
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.cell(0, 10, 'MRI BRAIN WITH CONTRAST', align='C', new_x="LMARGIN", new_y="NEXT")
        
        # Sub-title / Institution
        self.set_font('Helvetica', '', 10)
        self.set_text_color(100, 100, 100)  # Grey
        self.cell(0, 5, 'AI-Powered Neuroradiology Analysis System', align='C', new_x="LMARGIN", new_y="NEXT")
        
        # Horizontal separator line
        self.ln(5)
        self.set_draw_color(0, 51, 102)  # Dark blue
        self.set_line_width(0.8)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)
    
    def footer(self):
        """Professional footer with institution, disclaimer, and page number."""
        self.set_y(-22)
        
        # Layer 1 (Top): The Authority Anchor - Bold, Dark Grey
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(50, 50, 50)  # Dark grey (almost black)
        self.cell(0, 4, 'Project Undertaken at Rajagiri School of Engineering & Technology (RSET)', align='C', new_x="LMARGIN", new_y="NEXT")
        
        # Layer 2 (Middle): The Legal Safety Net - Italicized, Lighter Grey
        self.set_font('Helvetica', 'I', 7)
        self.set_text_color(130, 130, 130)  # Lighter grey
        self.cell(0, 4, 'DISCLAIMER: This report is AI-generated and requires review by a qualified radiologist.', align='C', new_x="LMARGIN", new_y="NEXT")
        
        # Layer 3 (Bottom): The Utility - Smallest font, centered
        self.set_font('Helvetica', '', 7)
        self.set_text_color(100, 100, 100)
        self.cell(0, 4, f'Page {self.page_no()}', align='C')
    
    def add_section_title(self, title: str):
        """Add a bold section header."""
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(0, 51, 102)  # Dark blue
        self.ln(4)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0)  # Reset to black
    
    def add_body_text(self, text: str):
        """Add normal paragraph text."""
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)  # Near black
        self.multi_cell(0, 5, text)
        self.ln(2)
    
    def add_findings_paragraph(self, text: str):
        """Add findings text with proper paragraph formatting."""
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 30, 30)
        
        # Split into paragraphs and handle each
        paragraphs = text.strip().split('\n\n')
        for i, para in enumerate(paragraphs):
            # Clean up whitespace
            para = ' '.join(para.split())
            if para:
                self.multi_cell(0, 5, para)
                if i < len(paragraphs) - 1:
                    self.ln(3)
        self.ln(2)
    
    def add_impression_item(self, number: int, text: str):
        """Add a numbered impression item."""
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(0)
        
        # Number
        self.cell(8, 6, f"{number}.", align='L')
        
        # Text (may wrap)
        self.set_font('Helvetica', '', 10)
        
        # Get current position for multi-line handling
        x = self.get_x()
        y = self.get_y()
        
        # Use multi_cell with offset
        self.set_xy(x, y)
        self.multi_cell(0, 5, text)
        self.ln(1)


def parse_report_sections(report_text: str) -> dict:
    """
    Parse the text report into sections.
    
    Args:
        report_text: The full report text
        
    Returns:
        Dictionary with section names as keys
    """
    sections = {
        'patient_id': '',
        'date': '',
        'clinical_indication': '',
        'technique': '',
        'comparison': '',
        'findings': '',
        'impression': [],
    }
    
    lines = report_text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        
        # Detect section headers
        if line_stripped.startswith('PATIENT ID:'):
            sections['patient_id'] = line_stripped.replace('PATIENT ID:', '').strip()
        elif line_stripped.startswith('DATE:'):
            sections['date'] = line_stripped.replace('DATE:', '').strip()
        elif line_stripped == 'CLINICAL INDICATION:':
            current_section = 'clinical_indication'
            current_content = []
        elif line_stripped == 'TECHNIQUE:':
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'technique'
            current_content = []
        elif line_stripped == 'COMPARISON:':
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'comparison'
            current_content = []
        elif line_stripped == 'FINDINGS:':
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'findings'
            current_content = []
        elif line_stripped == 'IMPRESSION:':
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = 'impression'
            current_content = []
        elif line_stripped.startswith('DISCLAIMER:'):
            if current_section and current_content:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = None
        elif current_section:
            current_content.append(line)
    
    # Handle impression as list
    if isinstance(sections['impression'], str):
        impression_text = sections['impression']
    else:
        impression_text = '\n'.join(current_content) if current_section == 'impression' else ''
    
    # Parse impression items
    impression_items = []
    for line in impression_text.split('\n'):
        line = line.strip()
        if line.startswith('1.') or line.startswith('2.') or line.startswith('3.'):
            # Remove the number prefix
            item_text = line[2:].strip()
            impression_items.append(item_text)
        elif line and impression_items:
            # Continuation of previous item
            impression_items[-1] += ' ' + line
    
    sections['impression'] = impression_items
    
    return sections


def generate_pdf_report(report_text: str, output_path: str, case_id: str = None):
    """
    Generate a professional PDF from report text.
    
    Args:
        report_text: The full report text
        output_path: Path to save the PDF
        case_id: Optional case ID override
    """
    # Parse sections
    sections = parse_report_sections(report_text)
    
    # Create PDF
    pdf = ProfessionalMRIReport()
    pdf.add_page()
    
    # =========================================================================
    # PATIENT DEMOGRAPHICS (Two-Column Layout)
    # =========================================================================
    pdf.set_font('Helvetica', 'B', 10)
    
    # Row 1: Patient ID (Left) vs Date (Right)
    pdf.cell(30, 6, "PATIENT ID:", align='L')
    pdf.set_font('Helvetica', '', 10)
    patient_id = case_id or sections['patient_id'] or 'Unknown'
    pdf.cell(65, 6, patient_id, align='L')
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(20, 6, "DATE:", align='L')
    pdf.set_font('Helvetica', '', 10)
    date = sections['date'] or datetime.now().strftime('%B %d, %Y')
    pdf.cell(0, 6, date, align='L', new_x="LMARGIN", new_y="NEXT")
    
    # Row 2: Referring Physician
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(30, 6, "REF. PHYSICIAN:", align='L')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(65, 6, "Referring Physician", align='L')
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(20, 6, "STATUS:", align='L')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, "AI-Assisted Draft", align='L', new_x="LMARGIN", new_y="NEXT")
    
    # Thin divider line
    pdf.ln(4)
    pdf.set_draw_color(200, 200, 200)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)
    
    # =========================================================================
    # CLINICAL INDICATION
    # =========================================================================
    pdf.add_section_title("CLINICAL INDICATION:")
    pdf.add_body_text(sections['clinical_indication'] or "Clinical indication not provided.")
    
    # =========================================================================
    # TECHNIQUE
    # =========================================================================
    pdf.add_section_title("TECHNIQUE:")
    pdf.add_body_text(sections['technique'] or "Standard multisequence MRI protocol.")
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    pdf.add_section_title("COMPARISON:")
    pdf.add_body_text(sections['comparison'] or "No prior imaging available for comparison.")
    
    pdf.ln(2)
    
    # =========================================================================
    # FINDINGS (Main Body)
    # =========================================================================
    pdf.add_section_title("FINDINGS:")
    pdf.add_findings_paragraph(sections['findings'] or "No findings documented.")
    
    pdf.ln(2)
    
    # =========================================================================
    # IMPRESSION (Highlighted)
    # =========================================================================
    pdf.add_section_title("IMPRESSION:")
    
    if sections['impression']:
        for i, item in enumerate(sections['impression'], 1):
            pdf.add_impression_item(i, item)
    else:
        pdf.add_body_text("No impression documented.")
    
    # =========================================================================
    # HARD FLOOR - Visual separation between clinical content and metadata
    # =========================================================================
    pdf.ln(12)
    pdf.set_draw_color(180, 180, 180)  # Light grey
    pdf.set_line_width(0.4)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())  # Full-width separator
    
    # Save PDF
    pdf.output(output_path)
    print(f"[OK] PDF Report saved: {output_path}")
    
    return output_path


def main():
    """Main entry point for PDF generation."""
    if len(sys.argv) < 2:
        print("Usage: python generate_pdf_report.py <results_folder>")
        print("Example: python generate_pdf_report.py results/BraTS-GLI-00009-000")
        sys.exit(1)
    
    results_folder = sys.argv[1]
    
    # Find the report file
    report_path = os.path.join(results_folder, 'feature_extraction', 'radiology_report.txt')
    
    if not os.path.exists(report_path):
        print(f"[ERROR] Report not found: {report_path}")
        print("Please generate the text report first using generate_report_gemini.py")
        sys.exit(1)
    
    # Read the report
    with open(report_path, 'r', encoding='utf-8') as f:
        report_text = f.read()
    
    # Extract case ID from folder name
    case_id = os.path.basename(results_folder.rstrip('/\\'))
    
    # Generate PDF path
    pdf_path = os.path.join(results_folder, 'feature_extraction', 'radiology_report.pdf')
    
    print("=" * 70)
    print("PROFESSIONAL PDF REPORT GENERATOR")
    print("=" * 70)
    print(f"Case ID: {case_id}")
    print(f"Input: {report_path}")
    print(f"Output: {pdf_path}")
    print("-" * 70)
    
    # Generate PDF
    generate_pdf_report(report_text, pdf_path, case_id)
    
    print("=" * 70)
    print("PDF GENERATION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
