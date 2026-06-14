# 1. Standard library imports
import os
import tempfile
import traceback
from functools import wraps
from pathlib import Path

# Load local .env file if it exist

env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip().strip('"').strip("'")

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

# Integrate with Gunicorn logging if running in production (WSGI)
if __name__ != '__main__':
    import logging
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

# Enable CORS for all origins because frontend calls ML service directly (with API Key validation)
CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})

# API Key security setup
ML_API_KEY = os.environ.get("ML_API_KEY", "")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        app.logger.info(f"🔑 API Key validation: received key length {len(key)}")
        if not ML_API_KEY:
            app.logger.warning("⚠️ API Key validation failed: ML_API_KEY is not configured in the environment variables.")
            return jsonify({"error": "Unauthorized"}), 401
        if key != ML_API_KEY:
            app.logger.warning(f"❌ API Key validation failed: key mismatch. Expected length {len(ML_API_KEY)} but received {len(key)}.")
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

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
@require_api_key
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

        # Group applications by jobId to avoid nested O(N*M) linear search
        apps_by_job = {}
        for app in applications:
            j_id = str(app.get('jobId', ''))
            if j_id:
                apps_by_job.setdefault(j_id, []).append(app)

        # Growth predictions
        predictions = {}
        for job in jobs:
            job_id = str(job.get('_id') or job.get('id') or '')
            job_apps = apps_by_job.get(job_id, [])
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
@require_api_key
def parse_resume_route():
    app.logger.info("📄 Received request to /parse_resume")
    if 'resume' not in request.files:
        app.logger.warning("❌ No resume file in request.files")
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['resume']
    if file.filename == '':
        app.logger.warning("❌ Resume filename is empty")
        return jsonify({'error': 'No selected file'}), 400

    # Validate file extension
    allowed_extensions = {'.pdf', '.docx', '.txt'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        app.logger.warning(f"❌ Unsupported file type extension: {file_ext}")
        return jsonify({'error': 'Unsupported file type'}), 400

    try:
        app.logger.info(f"📂 Saving uploaded file: {file.filename} (extension: {file_ext})")
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
            app.logger.error("❌ File saved but size is 0 or path does not exist")
            return jsonify({'error': 'File upload failed'}), 400

        app.logger.info(f"🔍 Parsing resume file of size: {os.path.getsize(temp_path)} bytes")
        parsed_data = parse_resume(temp_path)
        app.logger.info(f"✅ Resume parsed successfully. Extracted {len(parsed_data.get('skills', []))} skills.")
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
@require_api_key
def match_jobs():
    app.logger.info("💼 Received request to /match_jobs")
    try:
        data = request.json
        resume_skills = data.get('skills', [])
        jobs = data.get('jobs', [])
        app.logger.info(f"🔍 Matching {len(resume_skills)} resume skills against {len(jobs)} jobs")

        # Prepare batch of required skills
        job_list_skills = [job.get('requiredSkills', []) if isinstance(job, dict) else [] for job in jobs]
        
        # Calculate all scores at once
        scores = calculate_fit_batch(resume_skills, job_list_skills)

        results = []
        for i, job in enumerate(jobs):
            if not isinstance(job, dict):
                continue

            results.append({
                '_id': str(job.get('_id') or job.get('id') or 'N/A'),
                'id': str(job.get('id') or job.get('_id') or 'N/A'),
                'title': job.get('title', 'Untitled'),
                'description': job.get('description', ''),
                'requiredSkills': job.get('requiredSkills', []),
                'score': scores[i]
            })

        if not results:
            app.logger.warning("⚠️ No valid jobs found to match")
            return jsonify({'error': 'No valid jobs to match'}), 400

        results.sort(key=lambda x: x['score'], reverse=True)
        app.logger.info(f"✅ Successfully matched and ranked {len(results)} jobs")
        return jsonify({'matches': results})
    except Exception as e:
        app.logger.error(f"Job matching error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route("/")
def home():
    return jsonify({"message": "ML Project is running successfully!"})

# 10. Application entry point
if __name__ == '__main__':
    # Bind to PORT environment variable if available (Render compatibility), fallback to 5001
    port = int(os.environ.get("PORT", 5001))
    app.logger.info(f"Starting ML application on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
