# 1. Standard library imports
import re

# 2. Third-party imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 3. Skill normalization function with more comprehensive handling
def normalize_skill(skill):
    """Normalize skill names by removing spaces around special characters"""
    if not isinstance(skill, str):
        return ""
        
    skill = skill.lower().strip()
    skill = re.sub(r'\s+', ' ', skill)  # Collapse multiple spaces
    
    # Handle common skill variations
    variations = {
        r'c\s*#': 'c#',
        r'c\s*\+\+': 'c++',
        r'\.\s*net': '.net',
        r'\bjs\b': 'javascript',
        r'\bts\b': 'typescript',
        r'\bai\b': 'artificial intelligence',
        r'\bml\b': 'machine learning',
        r'\bcv\b': 'computer vision',
        r'\baws\b': 'amazon web services',
        r'\bgcp\b': 'google cloud platform'
    }
    
    for pattern, replacement in variations.items():
        skill = re.sub(pattern, replacement, skill)
    
    return skill

# 4. Job fit calculation with improved error handling
def calculate_fit_batch(resume_skills, job_list_skills):
    """Calculate job fit scores for a list of jobs in one batch operation"""
    if not resume_skills or not job_list_skills:
        return [0.0] * len(job_list_skills) if job_list_skills else []

    try:
        # Normalize resume skills once
        norm_resume_skills = list(set(normalize_skill(s) for s in resume_skills if s))
        if not norm_resume_skills:
            return [0.0] * len(job_list_skills)
        resume_text = ' '.join(norm_resume_skills)

        # Normalize all job skills
        job_texts = []
        for job_skills in job_list_skills:
            norm_job_skills = list(set(normalize_skill(s) for s in job_skills if s))
            job_texts.append(' '.join(norm_job_skills) if norm_job_skills else "")

        # Combine all texts for a single fit_transform
        all_texts = [resume_text] + job_texts
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between resume (index 0) and all jobs (index 1 onwards)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Convert to list of scores (0-100)
        return [max(0.0, min(100.0, round(score * 100, 1))) for score in similarities[0]]
    except Exception as e:
        print(f"Error in calculate_fit_batch: {e}")
        return [0.0] * len(job_list_skills)