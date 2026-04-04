# 1. Standard library imports
import os
import tempfile
import traceback
from pathlib import Path

# 2. Third-party imports
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import numpy as np

# 3. Local imports
from resume_parser import parse_resume
from job_matcher import calculate_fit_batch

# 4. Initialize Flask app
app = Flask(__name__)
CORS(app)

# 5. Configuration setup
BASE_DIR = Path(__file__).parent.resolve()
UPLOAD_FOLDER = BASE_DIR / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)  # Create uploads directory if not exists
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)

# 6. Resume download endpoint
@app.route('/uploads/<filename>', methods=['GET'])
def download_resume(filename):
    try:
        return send_from_directory(
            app.config['UPLOAD_FOLDER'], 
            filename, 
            as_attachment=True
        )
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 7. Analytics dashboard endpoint
@app.route('/analytics_dashboard', methods=['POST'])
def analytics_dashboard():
    try:
        data = request.json
        resumes = data.get('resumes', [])
        jobs = data.get('jobs', [])
        applications = data.get('applications', [])

        # Skill analysis
        skill_demand = {}
        skill_supply = {}
        for job in jobs:
            for skill in job.get('requiredSkills', []):
                skill_demand[skill] = skill_demand.get(skill, 0) + 1
        for resume in resumes:
            for skill in resume.get('skills', []):
                skill_supply[skill] = skill_supply.get(skill, 0) + 1

        # Fit score distribution
        fit_scores = [app.get('score', 0) for app in applications]
        histogram_bins = [0]*5
        for score in fit_scores:
            idx = min(int(score // 20), 4)
            histogram_bins[idx] += 1

        # Growth predictions
        predictions = {}
        for job in jobs:
            job_id = str(job.get('_id', ''))
            job_apps = [app for app in applications if app.get('jobId') == job_id]
            if len(job_apps) >= 2:
                try:
                    days = np.array([
                        (np.datetime64(app['date']) - np.datetime64(job['postedDate'])
                    ).astype(int) for app in job_apps]).reshape(-1, 1)
                    counts = np.arange(1, len(job_apps) + 1)
                    lr = LinearRegression().fit(days, counts)
                    future_days = np.array([[7], [14], [30]])
                    future_counts = lr.predict(future_days).tolist()
                    predictions[job_id] = {
                        "jobTitle": job.get('title', ''),
                        "predictedApplications": {
                            "7_days": max(0, future_counts[0]),
                            "14_days": max(0, future_counts[1]),
                            "30_days": max(0, future_counts[2])
                        }
                    }
                except Exception as e:
                    app.logger.error(f"Prediction error for job {job_id}: {e}")

        return jsonify({
            "skill_demand": skill_demand,
            "skill_supply": skill_supply,
            "fit_score_distribution": histogram_bins,
            "application_growth_predictions": predictions
        })
    except Exception as e:
        app.logger.error(f"Analytics error: {e}")
        return jsonify({'error': str(e)}), 500

# 8. Resume parsing endpoint with enhanced error handling
@app.route('/parse_resume', methods=['POST'])
def parse_resume_route():
    if 'resume' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Validate file extension
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        # Save to temporary file with proper extension
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            dir=app.config['UPLOAD_FOLDER']
        ) as tmp_file:
            file.save(tmp_file.name)
            temp_path = tmp_file.name

        # Verify file was saved correctly
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'File upload failed'}), 400

        parsed_data = parse_resume(temp_path)
        return jsonify(parsed_data)
    except Exception as e:
        app.logger.error(f"Resume parsing error: {e}\n{traceback.format_exc()}")
        return jsonify({
            'error': 'Failed to parse resume',
            'details': str(e)
        }), 500
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)

# 9. Job matching endpoint
@app.route('/match_jobs', methods=['POST'])
def match_jobs():
    try:
        data = request.json
        resume_skills = data.get('skills', [])
        jobs = data.get('jobs', [])

        # Prepare batch of required skills
        job_list_skills = [job.get('requiredSkills', []) if isinstance(job, dict) else [] for job in jobs]
        
        # Calculate all scores at once
        scores = calculate_fit_batch(resume_skills, job_list_skills)

        results = []
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                continue

            results.append({
                '_id': str(job.get('_id', 'N/A')),
                'title': job.get('title', 'Untitled'),
                'description': job.get('description', ''),
                'requiredSkills': job.get('requiredSkills', []),
                'score': scores[i]
            })

        if not results:
            return jsonify({'error': 'No valid jobs to match'}), 400

        results.sort(key=lambda x: x['score'], reverse=True)
        return jsonify({'matches': results})
    except Exception as e:
        app.logger.error(f"Job matching error: {e}")
        return jsonify({'error': str(e)}), 500

# 10. Application entry point
if __name__ == '__main__':
    # Production ready port, debug disabled for scalability
    app.run(host='0.0.0.0', port=5001, debug=False)



@app.route("/")
def home():
    return jsonify({"message": "ML Project is running successfully!"})    