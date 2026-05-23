# 1. Standard library imports
import os
import re
from pathlib import Path

# 2. Third-party imports
from docx import Document
import pdfplumber
import spacy
from spacy.matcher import PhraseMatcher
import spacy.cli

# 3. Try to load spaCy model or download if not found
try:
    # Disable parser, tagger, lemmatizer, and attribute_ruler to optimize speed/memory
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer", "attribute_ruler"])

# 4. Organization validation function
def is_valid_org(text):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return len(clean_text.strip().split()) <= 5 and len(clean_text.strip()) >= 2

# 5. Setup paths with absolute path resolution
BASE_DIR = Path(__file__).parent.resolve()
SKILLS_FILE = BASE_DIR / "skills_list.txt"

# 6. Verify skills file exists
if not SKILLS_FILE.exists():
    raise FileNotFoundError(f"Skills file not found at: {SKILLS_FILE}")

# 7. Load skills list and create mapping
with open(SKILLS_FILE, "r", encoding="utf-8") as f:
    skill_lines = [line.strip().lower() for line in f if line.strip()]
    
# 8. Create mapping of abbreviations to full forms
skill_map = {}
for skill in skill_lines:
    skill_map[skill] = skill
    if ' ' in skill:
        abbr = ''.join(word[0] for word in skill.split())
        skill_map[abbr] = skill

# 9. Create reverse mapping for lookup
reverse_skill_map = {}
for abbr, full in skill_map.items():
    reverse_skill_map[full] = full
    if abbr != full:
        reverse_skill_map[abbr] = full

# 10. Get unique skills for matching
skill_keywords = list(skill_map.keys())

# 11. Setup PhraseMatcher
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
patterns = [nlp.make_doc(skill) for skill in skill_keywords]
matcher.add("SKILLS", patterns)

# 12. Text extraction function with enhanced error handling
def extract_text(file_path):
    """Extract clean text from PDF, DOCX, or TXT"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    text = ""
    try:
        if str(file_path).endswith('.pdf'):
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        elif str(file_path).endswith('.docx'):
            doc = Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
    except Exception as e:
        raise RuntimeError(f"Error extracting text from {file_path}: {str(e)}")
    
    # Clean non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c in '\n\t ')
    return text

# 13. Main resume parsing function with improved error handling
def parse_resume(file_path):
    try:
        text = extract_text(file_path)
        
        # Fix common formatting issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'(\w)\s*([#+]\w*)', r'\1\2', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.lower()
        
        doc = nlp(text)

        skills = set()
        orgs = set()
        degrees = set()
        experience = 0

        # Entity extraction
        for ent in doc.ents:
            if ent.label_ == "ORG" and ent.text.isascii() and is_valid_org(ent.text):
                orgs.add(ent.text.strip())
            elif ent.label_ == "DATE" and any(char.isdigit() for char in ent.text):
                if 'year' in ent.text.lower() or 'yr' in ent.text.lower():
                    years = re.findall(r'\d+', ent.text)
                    if years:
                        experience += int(years[0])

        # Skill extraction
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            matched_text = span.text.lower()
            full_form = reverse_skill_map.get(matched_text, matched_text)
            skills.add(full_form)

        # Degree extraction
        education_keywords = {'bsc', 'msc', 'phd', 'bachelor', 'master', 'doctorate', 'bs', 'ba', 'ma', 'mba'}
        for token in doc:
            if token.text.lower() in education_keywords:
                degrees.add(token.text.lower())

        return {
            'skills': sorted(skills),
            'organizations': sorted(orgs),
            'degrees': sorted(degrees),
            'experience_years': experience
        }
    except Exception as e:
        raise RuntimeError(f"Failed to parse resume: {str(e)}")