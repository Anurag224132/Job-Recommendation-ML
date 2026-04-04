import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from job_matcher import calculate_fit_batch

def test_batch_performance():
    resume_skills = ["python", "machine learning", "flask", "aws", "docker", "kubernetes", "react"]
    
    # Simulate 500 jobs with varying matching skills
    base_job_skills = [
        ["python", "flask", "django"],
        ["react", "javascript", "typescript", "node.js"],
        ["aws", "devops", "kubernetes", "docker"],
        ["machine learning", "pytorch", "tensorflow", "python"],
        ["java", "spring boot", "microservices"]
    ]
    
    job_list_skills = base_job_skills * 100  # 500 jobs
    
    print(f"Starting batch matching for {len(job_list_skills)} jobs...")
    start_time = time.perf_counter()
    
    scores = calculate_fit_batch(resume_skills, job_list_skills)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"Batch matching completed in {duration:.4f} seconds.")
    print(f"Average time per job: {duration/len(job_list_skills):.6f} seconds.")
    print(f"First 5 scores: {scores[:5]}")
    
    # Assert we got the right number of scores
    assert len(scores) == len(job_list_skills)
    print("Verification successful!")

if __name__ == "__main__":
    test_batch_performance()
