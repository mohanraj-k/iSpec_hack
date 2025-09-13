import os
import logging
import pathlib
import subprocess
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import json
from datetime import datetime
import csv
import io
import re
import pandas as pd
from utils import mdd_csv_aggregator as mdd_agg

from utils.azure_openai_client import AzureOpenAIClient
from utils.numpy_matching_engine import NumpyMatchingEngine
from utils.enhanced_matching_engine import EnhancedMatchingEngine
from utils.mdd_output_generator import MDDOutputGenerator
from utils.mdd_file_processor import MDDFileProcessor
from utils.storage import storage
from utils.config import USE_S3
from utils.field_aliases import get_field_value as _gf
from utils.config import SESSION_SECRET, DEBUG, PORT, validate_config
logger = logging.getLogger(__name__)

# # Configure logging to write to Gunicorn's error log file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s %(levelname)s %(name)s: %(message)s',
#     handlers=[logging.FileHandler("gunicorn_error.log", mode="w"), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # Remove all handlers from root logger except FileHandler to silence terminal output
# root_logger = logging.getLogger()
# for handler in root_logger.handlers[:]:
#     if not isinstance(handler, logging.FileHandler):
#         root_logger.removeHandler(handler)


# Validate configuration (loads .env only in dev via utils.config)
validate_config()
# Create Flask app
app = Flask(__name__)
app.secret_key = SESSION_SECRET
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Align Flask/root loggers with Gunicorn's logging when running under Gunicorn
# This makes sure our app logs go to stdout/stderr via Gunicorn so Grafana/K8s can collect them.
gunicorn_logger = logging.getLogger("gunicorn.error")
if gunicorn_logger.handlers:
    # Use Gunicorn's handlers/level for Flask app logger
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)
    app.logger.propagate = False

    # Point root logger to Gunicorn as well so all module loggers propagate
    root = logging.getLogger()
    root.handlers = gunicorn_logger.handlers
    root.setLevel(gunicorn_logger.level)

# Tame noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


import uuid
from flask import jsonify, request

# --- Progress Bar Backend Implementation ---
progress_dict = {}

# In-memory store of Target MDD rows per upload session
# Structure: { upload_id: { dq_name_lower: row_dict } }
TARGET_MDD_ROWS = {}

def set_progress(upload_id, percent, message):
    progress_dict[upload_id] = {"percent": percent, "message": message}

def get_progress(upload_id):
    return progress_dict.get(upload_id, {"percent": 0, "message": "Starting..."})

@app.route("/progress/<upload_id>")
def progress(upload_id):
    progress_data = get_progress(upload_id)
    return jsonify(progress_data)

# Test endpoint for manual progress bar check
@app.route("/progress-test/<upload_id>")
def progress_test(upload_id):
    logger.info(f"[PROGRESS-TEST] /progress-test/{upload_id} called")
    progress_data = get_progress(upload_id)
    logger.info(f"[PROGRESS-TEST] Returning: {progress_data}")
    return jsonify(progress_data)

# Configuration
MAX_UPLOAD_MB = 20  # Hardcoded max upload size (in MB)
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

# Ensure required directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('output', exist_ok=True)
os.makedirs('MDD_DATABASE', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs(os.path.join('output', 'dq_scripts'), exist_ok=True)

# Initialize components
azure_client = AzureOpenAIClient()
file_processor = MDDFileProcessor()
# enhanced_matching_engine = NumpyMatchingEngine(azure_client)
enhanced_matching_engine = EnhancedMatchingEngine(azure_client)
mdd_generator = MDDOutputGenerator()

# Auto-load precomputed embeddings at startup
try:
    logger.info("Attempting to load precomputed vector database...")
    if enhanced_matching_engine.initialize_vector_db():
        logger.info("Vector database initialized successfully")
    else:
        logger.warning("No precomputed vector database found")
except Exception as e:
    logger.error(f"Failed to load vector database: {str(e)}")

from utils.match_thresholds import THRESHOLDS


def _slugify_filename(name: str, max_len: int = 60) -> str:
    """Create a filesystem-safe slug for filenames."""
    import re
    name = (name or "dq_script").strip()
    # Replace spaces and invalids with underscore
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    # Collapse repeats
    name = re.sub(r"_+", "_", name)
    name = name.strip("_.-")
    if not name:
        name = "dq_script"
    if len(name) > max_len:
        name = name[:max_len]
    return name


@app.route('/generate-dq-script', methods=['POST'])
def generate_dq_script():
    """Generate a Python DQ script file for a given row and return a download URL."""
    try:
        data = request.get_json(silent=True) or {}
        dq_name = data.get('dq_name') or data.get('DQ Name') or data.get('Check Name') or 'DQ_Script'
        tech_code = data.get('tech_code') or data.get('Pseudo Tech Code (Copy Source Study)') or ''
        dq_description = data.get('dq_description') or data.get('Target Check Description') or ''
        query_text = data.get('query_text') or data.get('Target Query Text') or ''
        ctx = data.get('domain_context') or {}
        upload_id = data.get('upload_id') or ''

        # Prefer domain context from Target MDD (by upload_id + dq_name)
        try:
            if upload_id and dq_name and TARGET_MDD_ROWS.get(upload_id):
                _map = TARGET_MDD_ROWS.get(upload_id) or {}
                _row = _map.get(str(dq_name).strip().lower())
                if _row:
                    logger.info(f'Rs-line164 generate_dq_script : _row {_row}')
                    # Build context from Target MDD row
                    _domain = _gf(_row, 'primary_dataset')
                    _form = _gf(_row, 'P_form_name')
                    _visit = _gf(_row, 'P_visit_name')
                    _vars = _gf(_row, 'primary_dataset_columns')
                    logger.info(f'Rs-line166 generate_dq_script : _domain {_domain}, _form {_form}, _visit {_visit}, _vars {_vars}')
                    # Aggregate relational pieces
                    _rel_domains = []
                    _rel_vars = []
                    _rel_dyn = []
                    for _k in ('R1_Domain', 'R2_Domain', 'R3_Domain', 'R4_Domain', 'R5_Domain'):
                        _v = _gf(_row, _k)
                        if _v:
                            _rel_domains.append(_v)
                    for _k in ('R1_Domain_Variables', 'R2_Domain_Variables', 'R3_Domain_Variables', 'R4_Domain_Variables', 'R5_Domain_Variables'):
                        _v = _gf(_row, _k)
                        if _v:
                            _rel_vars.append(_v)
                    for _k in ('R1_Dynamic_Columns', 'R2_Dynamic_Columns', 'R3_Dynamic_Columns', 'R4_Dynamic_Columns', 'R5_Dynamic_Columns'):
                        _v = _gf(_row, _k)
                        if _v:
                            _rel_dyn.append(_v)

                    ctx = {
                        'domain': _domain,
                        'form': _form,
                        'visit': _visit,
                        'variables': _vars,
                        'relational_domains': ', '.join(_rel_domains) if _rel_domains else '',
                        'relational_variables': ', '.join(_rel_vars) if _rel_vars else '',
                        'relational_dynamic_variables': ', '.join(_rel_dyn) if _rel_dyn else '',
                    }
        except Exception as _ctx_ex:
            logger.warning(f"Failed building domain_context from Target MDD for upload_id={upload_id}, dq_name={dq_name}: {_ctx_ex}")

        # Prepare domain context from loose keys if not provided
        if not ctx:
            ctx = {
                'domain': data.get('domain') or data.get('Domain') or '',
                'form': data.get('form') or data.get('Form Name') or '',
                'visit': data.get('visit') or data.get('Visit Name') or '',
                'variables': data.get('variables') or data.get('Primary Domain Variables (Pre-Conf)') or '',
            }

        # Generate script content
        logger.info(f'Rs-line208 generate_dq_script : {dq_name},{dq_description},{query_text},{tech_code},{ctx}')
        script_content = azure_client.generate_python_script(
            dq_name=dq_name,
            check_description=dq_description,
            query_text=query_text,
            tech_code=tech_code,
            domain_context=ctx,
        )

        # Build filename
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        base = _slugify_filename(dq_name)
        rel_path = f"dq_scripts/{base}_{ts}.py"

        # Persist file
        if USE_S3:
            key = f"{app.config['OUTPUT_FOLDER']}/{rel_path}"
            storage.write_bytes(key, script_content.encode('utf-8'))
        else:
            out_dir = os.path.join(app.config['OUTPUT_FOLDER'], 'dq_scripts')
            os.makedirs(out_dir, exist_ok=True)
            full_path = os.path.join(out_dir, f"{base}_{ts}.py")
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(script_content)

        return jsonify({
            'success': True,
            'filename': rel_path,
            'download_url': f"/download/{rel_path}",
        })
    except Exception as e:
        logger.error(f"Error generating DQ script: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def index():
    """Main application page"""
    return render_template('index.html', thresholds=THRESHOLDS, max_upload_mb=MAX_UPLOAD_MB)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Generate or get upload_id
    logger.info("[UPLOAD] Received upload request")
    upload_id = request.form.get("upload_id") or uuid.uuid4().hex
    set_progress(upload_id, 10, "Uploading file...")
    logger.info(f"[UPLOAD] Started upload_file for upload_id={upload_id}")
    try:
        if 'target_file' not in request.files:
            logger.error(f"[UPLOAD] No file part in request for upload_id={upload_id}")
            return jsonify({'error': 'No file selected', 'upload_id': upload_id}), 400
        
        file = request.files['target_file']
        if file.filename == '':
            logger.error(f"[UPLOAD] Empty filename in request for upload_id={upload_id}")
            return jsonify({'error': 'No file selected', 'upload_id': upload_id}), 400
        
        if not file.filename or not file.filename.lower().endswith(('.xlsx', '.csv')):
            logger.error(f"[UPLOAD] Invalid file type: {file.filename} for upload_id={upload_id}")
            return jsonify({'error': 'Invalid file type. Only .xlsx or .csv files are supported.', 'upload_id': upload_id}), 400
        
        # Save uploaded file (local or S3)
        filename = secure_filename(file.filename)
        if not filename:
            logger.error(f"[UPLOAD] Invalid filename after secure_filename for upload_id={upload_id}")
            return jsonify({'error': 'Invalid filename'}), 400

        key = f"uploads/{filename}"
        logger.info(f"[UPLOAD] Saving file to storage: {key} for upload_id={upload_id}")
        file_content = file.read()
        storage.write_bytes(key, file_content)

        # For downstream processing (process_mdd_file expects a local path)
        if USE_S3:
            import tempfile
            temp_dir = pathlib.Path("/tmp/uploads")
            temp_dir.mkdir(parents=True, exist_ok=True)
            filepath = temp_dir / filename
            filepath.write_bytes(file_content)
            filepath = str(filepath)
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            pathlib.Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(file_content)

        logger.info(f"[UPLOAD] File stored and local path ready: {filepath} for upload_id={upload_id}")
        set_progress(upload_id, 20, "File uploaded. Starting processing...")
        
        try:
            sponsor_name = request.form.get('sponsor_name', '').strip()
            match_scope = (request.form.get('match_scope', 'across') or 'across').strip().lower()
            logger.info(f"[UPLOAD] Calling process_mdd_file for upload_id={upload_id} with sponsor_name={sponsor_name} match_scope={match_scope}")
            result = process_mdd_file(filepath, upload_id=upload_id, sponsor_name=sponsor_name, match_scope=match_scope)
            logger.info(f"[UPLOAD] process_mdd_file returned type={type(result)} value={repr(result)[:500]} for upload_id={upload_id}")
            # If processing reported an error in its payload, surface it as an HTTP error
            if isinstance(result, dict) and result.get('error'):
                err_msg = str(result.get('error'))
                set_progress(upload_id, 100, f"Processing failed: {err_msg}")
                logger.info(f"[UPLOAD] Returning processing error for upload_id={upload_id}: {err_msg}")
                return jsonify({'error': err_msg, 'upload_id': upload_id}), 400

            set_progress(upload_id, 100, "Processing completed successfully!")
            logger.info(f"[UPLOAD] Preparing success response for upload_id={upload_id}")
            response = {'success': True, 'result': result, 'upload_id': upload_id}
            logger.info(f"[UPLOAD] Response content: {repr(response)[:500]} for upload_id={upload_id}")
            return jsonify(response)
        except Exception as process_error:
            logger.error(f"[UPLOAD] File processing error: {str(process_error)} for upload_id={upload_id}")
            logger.error(f"[UPLOAD] Error type: {type(process_error).__name__} for upload_id={upload_id}")
            import traceback
            logger.error(f"[UPLOAD] Traceback: {traceback.format_exc()} for upload_id={upload_id}")
            set_progress(upload_id, 100, f"File processing failed: {str(process_error)}")
            logger.info(f"[UPLOAD] Preparing error response for upload_id={upload_id}")
            return jsonify({'error': f'File processing failed: {str(process_error)}', 'upload_id': upload_id}), 500
        
    except Exception as e:
        logger.error(f"[UPLOAD] Upload error: {str(e)} for upload_id={upload_id}")
        import traceback
        logger.error(f"[UPLOAD] Traceback: {traceback.format_exc()} for upload_id={upload_id}")
        set_progress(upload_id, 100, f"Upload failed: {str(e)}")
        logger.info(f"[UPLOAD] Preparing upload error response for upload_id={upload_id}")
        return jsonify({'error': f'Upload failed: {str(e)}', 'upload_id': upload_id}), 500

# -------------------------------------------------------------
# NEW ENDPOINT: Upload reference MDD files to database folder
# -------------------------------------------------------------
@app.route('/upload-mdd', methods=['POST'])
def upload_mdd_files():
    """Upload one or more reference MDD Excel files into the MDD_DATABASE folder"""
    try:
        if 'db_files' not in request.files:
            return jsonify({'success': False, 'error': 'No files part in request'}), 400

        files = request.files.getlist('db_files')
        if not files:
            return jsonify({'success': False, 'error': 'No files selected'}), 400

        db_path = os.path.join(os.getcwd(), 'MDD_DATABASE')
        os.makedirs(db_path, exist_ok=True)

        saved_files = []
        for f in files:
            if f.filename == '':
                continue
            if not f.filename.lower().endswith(('.xlsx', '.csv')):
                return jsonify({'success': False, 'error': 'Invalid file type. Only .xlsx or .csv allowed.'}), 400
            filename = secure_filename(f.filename)
            # Write to storage (S3 or local)
            key = f"MDD_DATABASE/{filename}"
            file_bytes = f.read()
            storage.write_bytes(key, file_bytes)
            # For local dev, also ensure a local copy exists in MDD_DATABASE
            if not USE_S3:
                dest = os.path.join(db_path, filename)
                with open(dest, 'wb') as lf:
                    lf.write(file_bytes)
            saved_files.append(filename)

        logger.info(f"Uploaded {len(saved_files)} file(s) to MDD_DATABASE: {saved_files}")
        return jsonify({'success': True, 'message': f'Uploaded {len(saved_files)} file(s) successfully', 'files': saved_files})
    except Exception as e:
        logger.error(f"Error uploading MDD files: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/rebuild-database-start', methods=['POST'])
def rebuild_database_start():
    """Start vector database rebuild in background and report progress via /progress/<id>."""
    try:
        logger.info("Starting background rebuild with progress tracking...")
        import threading
        rebuild_id = uuid.uuid4().hex
        set_progress(rebuild_id, 1, "Preparing rebuild...")

        # Discover MDD files (same logic as synchronous route)
        mdd_files = []
        if USE_S3:
            import pathlib
            from utils.storage import storage as _storage
            tmp_dir = pathlib.Path('/tmp/MDD_DATABASE')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for key, _m in _storage.list_keys(prefix='MDD_DATABASE'):
                if str(key).lower().endswith(('.xlsx', '.csv')):
                    fname = str(key).split('/')[-1]
                    local = tmp_dir / fname
                    try:
                        local.write_bytes(_storage.read_bytes(key))
                        mdd_files.append(str(local))
                    except Exception as ex:
                        logger.warning(f"Failed to mirror {key} for rebuild: {ex}")
        else:
            mdd_database_path = os.path.join(os.getcwd(), 'MDD_DATABASE')
            if os.path.exists(mdd_database_path):
                for file in os.listdir(mdd_database_path):
                    if file.endswith(('.xlsx', '.csv')):
                        mdd_files.append(os.path.join(mdd_database_path, file))

        if not mdd_files:
            set_progress(rebuild_id, 100, 'No MDD files found in MDD_DATABASE directory')
            return jsonify({'success': False, 'error': 'No MDD files found in MDD_DATABASE directory', 'rebuild_id': rebuild_id}), 400

        def progress_cb(processed: int, total: int, message: str):
            try:
                total = max(total, 1)
                base = 10  # Reserve first 10% for preparation
                span = 85  # Use up to 95% for embedding generation
                percent = base + int(span * (processed / total))
                if percent >= 99:
                    percent = 99  # Leave headroom for finalization
                set_progress(rebuild_id, percent, message)
            except Exception:
                pass

        def run_rebuild():
            try:
                set_progress(rebuild_id, 10, f"Found {len(mdd_files)} files. Parsing and preparing...")
                ok = enhanced_matching_engine.append_vector_database(mdd_files, progress_cb=progress_cb)
                if ok:
                    stats = enhanced_matching_engine.get_statistics()
                    total_vec = stats.get('total_vectors', 0)
                    set_progress(rebuild_id, 100, f"Rebuild completed. Total vectors: {total_vec}")
                else:
                    set_progress(rebuild_id, 100, "Rebuild failed. Check server logs.")
            except Exception as e:
                logger.error(f"Background rebuild error: {str(e)}")
                set_progress(rebuild_id, 100, f"Rebuild failed: {str(e)}")

        threading.Thread(target=run_rebuild, daemon=True).start()
        return jsonify({'success': True, 'rebuild_id': rebuild_id})
    except Exception as e:
        logger.error(f"Error starting background rebuild: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/initialize-database', methods=['POST'])
def initialize_database():
    """Initialize the text-based matching database from MDD_DATABASE files"""
    try:
        logger.info("Initializing reference database...")
        
        # Load and process all MDD database files
        database_files = []
        database_dir = 'MDD_DATABASE'
        if USE_S3:
            import pathlib
            tmp_dir = pathlib.Path('/tmp/MDD_DATABASE')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for key, _mtime in storage.list_keys(prefix=f"{database_dir}"):
                if key.lower().endswith(('.xlsx', '.csv')):
                    filename = key.split('/')[-1]
                    local_path = tmp_dir / filename
                    local_path.write_bytes(storage.read_bytes(key))
                    database_files.append(str(local_path))
        else:
            if not os.path.exists(database_dir):
                return jsonify({'error': 'MDD_DATABASE directory not found'}), 400
            for filename in os.listdir(database_dir):
                if filename.lower().endswith(('.xlsx', '.csv')):
                    filepath = os.path.join(database_dir, filename)
                    database_files.append(filepath)
        
        if not database_files:
            return jsonify({'error': 'No Excel files found in MDD_DATABASE directory'}), 400
        
        # Start background build using engine API (storage-aware)
        import threading
        def build_complete_database():
            try:
                logger.info("Starting complete embeddings database build via engine API...")
                ok = enhanced_matching_engine.rebuild_vector_database(database_files)
                if ok:
                    logger.info("Complete database build finished successfully")
                    enhanced_matching_engine.load_precomputed_embeddings()
                else:
                    logger.error("Database build failed in engine API")
            except Exception as e:
                import traceback
                logger.error(f"Database build error: {str(e)}\n{traceback.format_exc()}")
        threading.Thread(target=build_complete_database, daemon=True).start()
        
        return jsonify({
            'success': True,
            'message': f'Building complete OpenAI embeddings database from Library of DQ specifications',
            'status': 'Processing all 722 records with OpenAI embeddings...',
            'files_to_process': len(database_files),
            'expected_records': 722
        })
            
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        return jsonify({'error': f'Database initialization failed: {str(e)}'}), 500

@app.route('/database-summary')
def database_summary():
    """Get reference database summary information"""
    try:
        # Try to load precomputed embeddings first
        if not enhanced_matching_engine.is_initialized():
            logger.info("Attempting to load precomputed embeddings...")
            loaded = enhanced_matching_engine.load_precomputed_embeddings()
            if not loaded:
                return jsonify({
                    'error': 'Reference database not initialized',
                    'initialized': False
                })
        
        stats = enhanced_matching_engine.get_statistics()
        
        # Safely get metadata if available
        metadata = getattr(enhanced_matching_engine, 'metadata', [])
        
        # Group records by source file
        file_breakdown = {}
        for record in metadata:
            source_file = record.get('source_file', 'Unknown')
            # logger.info(f"Rs-line317 - logging source_file: {source_file}")
            if source_file not in file_breakdown:
                file_breakdown[source_file] = 0
            file_breakdown[source_file] += 1
        
        # Get sample records (first 5)
        sample_records = []
        for i, record in enumerate(metadata[:5]):
            # Safely get field values
            dq_desc = record.get('DQ Description', record.get('Check Description', record.get('EC Description', 'N/A')))
            query_text = record.get('Standard Query text', record.get('Query Text', 'N/A'))
            
            # Truncate text if too long
            if isinstance(dq_desc, str) and len(dq_desc) > 100:
                dq_desc = dq_desc[:100] + '...'
            if isinstance(query_text, str) and len(query_text) > 100:
                query_text = query_text[:100] + '...'
            
            sample = {
                'dq_description': str(dq_desc),
                'query_text': str(query_text),
                'source_file': record.get('source_file', 'Unknown')
            }
            sample_records.append(sample)
        
        # Calculate file sizes
        import os
        database_size = {}
        try:
            if os.path.exists('data/faiss_index.bin'):
                index_size = os.path.getsize('data/faiss_index.bin') / (1024 * 1024)  # MB
                database_size['index_mb'] = round(index_size, 1)
            else:
                database_size['index_mb'] = 0.0
                
            if os.path.exists('data/metadata.pkl'):
                metadata_size = os.path.getsize('data/metadata.pkl') / 1024  # KB
                database_size['metadata_kb'] = round(metadata_size, 1)
            else:
                database_size['metadata_kb'] = 0.0
        except Exception:
            database_size = {'index_mb': 0.0, 'metadata_kb': 0.0}
        
        # Extract sponsor information from file names
        sponsor_breakdown = {}
        domain_breakdown = {}
        for filename, count in file_breakdown.items():
            # Extract sponsor from filename
            if 'Abbvie' in filename:
                sponsor = 'Abbvie'
                domain = 'Oncology'
            elif 'Astex' in filename:
                sponsor = 'Astex'
                domain = 'Oncology'
            elif 'AZ' in filename:
                sponsor = 'AZ'
                domain = 'Oncology'
            elif 'Cytokinetics' in filename:
                sponsor = 'Cytokinetics'
                domain = 'Cardiovascular'
            elif 'Kura' in filename:
                sponsor = 'Kura'
                domain = 'Oncology'
            else:
                sponsor = 'Unknown'
                domain = 'Unknown'
                
            sponsor_breakdown[sponsor] = sponsor_breakdown.get(sponsor, 0) + count
            domain_breakdown[domain] = domain_breakdown.get(domain, 0) + count

        # Calculate correct totals from file_breakdown
        total_records = sum(file_breakdown.values())
        total_files = len(file_breakdown)
        
        summary = {
            'initialized': True,
            'total_records': total_records,
            'total_files': total_files,
            'valid_embeddings': total_records,
            'embeddings_dimension': stats.get('embedding_dimension', 1536),
            'file_breakdown': file_breakdown,
            'sponsor_breakdown': sponsor_breakdown,
            'domain_breakdown': domain_breakdown,
            'database_size': database_size,
            'sample_records': sample_records
        }
        
        return jsonify(summary)
        
    except Exception as e:
        logger.error(f"Error getting database summary: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': 'Database not available', 'initialized': False}), 500

@app.route('/rebuild-database', methods=['POST'])
def rebuild_database():
    """Rebuild vector database with current MDD files"""
    try:
        logger.info("Starting dynamic vector database rebuild...")
        
        # Get list of MDD files from storage
        mdd_files = []
        if USE_S3:
            import pathlib
            from utils.storage import storage as _storage
            tmp_dir = pathlib.Path('/tmp/MDD_DATABASE')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            for key, _m in _storage.list_keys(prefix='MDD_DATABASE'):
                if key.lower().endswith(('.xlsx', '.csv')):
                    fname = key.split('/')[-1]
                    local = tmp_dir / fname
                    local.write_bytes(_storage.read_bytes(key))
                    mdd_files.append(str(local))
        else:
            mdd_database_path = os.path.join(os.getcwd(), 'MDD_DATABASE')
            if not os.path.exists(mdd_database_path):
                return jsonify({
                    'success': False,
                    'error': 'MDD_DATABASE directory not found'
                }), 400
            for file in os.listdir(mdd_database_path):
                if file.endswith(('.xlsx', '.csv')):
                    mdd_files.append(os.path.join(mdd_database_path, file))
        
        if not mdd_files:
            return jsonify({
                'success': False,
                'error': 'No MDD files found in MDD_DATABASE directory'
            }), 400
        
        logger.info(f"Found {len(mdd_files)} MDD files for rebuilding")
        
        # Append new records (if any) to existing database
        success = enhanced_matching_engine.append_vector_database(mdd_files)
        
        if success:
            stats = enhanced_matching_engine.get_statistics()
            
            # Get updated file breakdown
            metadata = getattr(enhanced_matching_engine, 'metadata', [])
            file_breakdown = {}
            for record in metadata:
                source_file = record.get('source_file', 'Unknown')
                if source_file not in file_breakdown:
                    file_breakdown[source_file] = 0
                file_breakdown[source_file] += 1
            
            total_records = sum(file_breakdown.values())
            total_files = len(file_breakdown)
            
            logger.info(f"Vector database updated successfully with {total_records} records from {total_files} files")
            
            return jsonify({
                'success': True,
                'message': f'Vector database updated successfully with {total_records} records from {total_files} files',
                'total_records': total_records,
                'total_files': total_files,
                'file_breakdown': file_breakdown
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to rebuild vector database'
            }), 500
    except Exception as e:
        logger.error(f"Error rebuilding database: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Duplicate simple background rebuild route removed in favor of progress-tracked route above

@app.route('/rebuild-check-logic-mdd', methods=['POST'])
def rebuild_check_logic_mdd():
    """Aggregate Check Logic MDD in-process and persist CSV (S3 in prod, local in dev)."""
    try:
        root_dir = os.getcwd()
        if USE_S3:
            prefix = 'Check_logic_collated_MDD/Source_MDDs/'
            tmp_dir = pathlib.Path('/tmp/Check_logic_Source_MDDs')
            tmp_dir.mkdir(parents=True, exist_ok=True)
            try:
                keys = storage.list_keys(prefix=prefix)
            except Exception as ex:
                logger.error("Failed to list S3 keys for %s: %s", prefix, str(ex))
                keys = []
            # Mirror only CSV files (case-insensitive)
            for key, _mtime in keys:
                try:
                    if not str(key).lower().endswith('.csv'):
                        continue
                    rel = key[len(prefix):] if str(key).startswith(prefix) else os.path.basename(str(key))
                    local_path = tmp_dir / rel
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    local_path.write_bytes(storage.read_bytes(key))
                except Exception as ex:
                    logger.warning("Failed to mirror %s from S3: %s", key, str(ex))
            input_root = tmp_dir
        else:
            input_root = pathlib.Path(root_dir) / 'Check_logic_collated_MDD' / 'Source_MDDs'
        logger.info("Starting Check Logic aggregation from %s", str(input_root))

        # Build aggregated DataFrame
        df = mdd_agg.aggregate_csvs(input_root)
        rows = len(df)
        logger.info("Aggregated %d rows of Check Logic", rows)

        # Persist CSV
        csv_bytes = df.to_csv(index=False).encode('utf-8')
        output_rel = 'Check_logic_collated_MDD/Output_ref_mdd/aggregated_checks.csv'
        if USE_S3:
            storage.write_bytes(output_rel, csv_bytes)
            output_path = f"s3://{output_rel}"
        else:
            output_path = os.path.join(root_dir, output_rel)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                f.write(csv_bytes)

        # Invalidate cached loader so UI can immediately see updates
        try:
            if hasattr(load_aggregated_check_logic, '_cache_sig'):
                delattr(load_aggregated_check_logic, '_cache_sig')
        except Exception:
            pass

        return jsonify({
            'success': True,
            'message': 'Check_logic MDD aggregated successfully',
            'output_file': output_path,
            'rows': rows,
        })
    except Exception as e:
        logger.error(f"Error running Check_logic aggregator in-process: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process-status/<task_id>')
def process_status(task_id):
    """Check processing status (placeholder for future async processing)"""
    return jsonify({'status': 'completed', 'progress': 100})

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download generated files"""
    try:
        if USE_S3:
            import io
            key = f"{app.config['OUTPUT_FOLDER']}/{filename}"
            try:
                data = storage.read_bytes(key)
            except Exception:
                return jsonify({'error': 'File not found'}), 404
            mimetype = 'application/octet-stream'
            if filename.endswith('.json'):
                mimetype = 'application/json'
            elif filename.endswith('.csv'):
                mimetype = 'text/csv'
            return send_file(io.BytesIO(data), as_attachment=True, download_name=filename, mimetype=mimetype)
        else:
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.exists(filepath):
                if filename.endswith('.json'):
                    return send_file(filepath, as_attachment=True, download_name=filename, mimetype='application/json')
                elif filename.endswith('.csv'):
                    return send_file(filepath, as_attachment=True, download_name=filename, mimetype='text/csv')
                else:
                    return send_file(filepath, as_attachment=True)
            else:
                return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/view-script/<path:filename>')
def view_script(filename):
    """Render a generated script inline in the browser as text/plain.

    Only allows files under output/dq_scripts/ and extensions .py or .txt
    """
    try:
        # Basic allow list
        if not (filename.startswith('dq_scripts/') and (filename.endswith('.py') or filename.endswith('.txt'))):
            return jsonify({'error': 'Unsupported file'}), 400

        if USE_S3:
            key = f"{app.config['OUTPUT_FOLDER']}/{filename}"
            try:
                data = storage.read_bytes(key)
            except Exception:
                return jsonify({'error': 'File not found'}), 404
            from flask import Response
            return Response(data, mimetype='text/plain; charset=utf-8')
        else:
            output_dir = os.path.join(os.getcwd(), app.config['OUTPUT_FOLDER'], 'dq_scripts')
            # Build absolute path and prevent traversal
            full_path = os.path.abspath(os.path.join(os.getcwd(), app.config['OUTPUT_FOLDER'], filename))
            if not full_path.startswith(os.path.abspath(output_dir) + os.sep):
                return jsonify({'error': 'Invalid path'}), 400
            if not os.path.exists(full_path):
                return jsonify({'error': 'File not found'}), 404
            return send_file(full_path, as_attachment=False, download_name=os.path.basename(full_path), mimetype='text/plain')
    except Exception as e:
        logger.error(f"View script error: {str(e)}")
        return jsonify({'error': f'Failed to view script: {str(e)}'}), 500

@app.route('/view-json/<filename>')
def view_json(filename):
    """View JSON file content directly"""
    try:
        if USE_S3:
            key = f"{app.config['OUTPUT_FOLDER']}/{filename}"
            if not filename.endswith('.json'):
                return jsonify({'error': 'JSON file not found'}), 404
            import json as _json
            try:
                data = _json.loads(storage.read_bytes(key).decode('utf-8'))
            except Exception:
                return jsonify({'error': 'JSON file not found'}), 404
            return jsonify(data)
        else:
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.exists(filepath) and filename.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return jsonify(data)
            else:
                return jsonify({'error': 'JSON file not found'}), 404
    except Exception as e:
        logger.error(f"JSON view error: {str(e)}")
        return jsonify({'error': f'Failed to view JSON: {str(e)}'}), 500

@app.route('/latest-results')
def latest_results():
    """Get the latest processing results as JSON"""
    try:
        # Find the most recent results file (matches only preferred)
        if USE_S3:
            keys = storage.list_keys(prefix=f"{app.config['OUTPUT_FOLDER']}", suffix='.json')
            if not keys:
                return jsonify({'error': 'No results files found'}), 404
            # Prefer matches_only
            matches = [(k, m) for (k, m) in keys if 'matches_only' in k]
            cand = matches if matches else keys
            cand.sort(key=lambda x: x[1], reverse=True)
            latest_key = cand[0][0]
            latest_file = latest_key.split('/')[-1]
            import json as _json
            data = _json.loads(storage.read_bytes(latest_key).decode('utf-8'))
        else:
            output_dir = app.config['OUTPUT_FOLDER']
            matches_files = [f for f in os.listdir(output_dir) if 'matches_only' in f and f.endswith('.json')]
            if matches_files:
                latest_file = sorted(matches_files, reverse=True)[0]
            else:
                json_files = [f for f in os.listdir(output_dir) if f.endswith('_results_') and f.endswith('.json')]
                if not json_files:
                    return jsonify({'error': 'No results files found'}), 404
                latest_file = sorted(json_files, reverse=True)[0]
            filepath = os.path.join(output_dir, latest_file)
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        return jsonify({
            'filename': latest_file,
            'data': data
        })
    except Exception as e:
        logger.error(f"Latest results error: {str(e)}")
        return jsonify({'error': f'Failed to get latest results: {str(e)}'}), 500

@app.route('/download-latest-json')
def download_latest_json():
    """Direct download of latest JSON results"""
    try:
        if USE_S3:
            import io
            keys = storage.list_keys(prefix=f"{app.config['OUTPUT_FOLDER']}", suffix='.json')
            if not keys:
                return jsonify({'error': 'No JSON files found'}), 404
            keys.sort(key=lambda x: x[1], reverse=True)
            latest_key = keys[0][0]
            latest_file = latest_key.split('/')[-1]
            data = storage.read_bytes(latest_key)
            return send_file(io.BytesIO(data), as_attachment=True, download_name=latest_file, mimetype='application/json')
        else:
            output_dir = app.config['OUTPUT_FOLDER']
            json_files = []
            for f in os.listdir(output_dir):
                if f.endswith('.json'):
                    filepath = os.path.join(output_dir, f)
                    mtime = os.path.getmtime(filepath)
                    json_files.append((mtime, f))
            if not json_files:
                return jsonify({'error': 'No JSON files found'}), 404
            json_files.sort(reverse=True)
            latest_file = json_files[0][1]
            filepath = os.path.join(output_dir, latest_file)
            return send_file(filepath, as_attachment=True, download_name=latest_file, mimetype='application/json')
    except Exception as e:
        logger.error(f"Download latest JSON error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

def extract_target_forms(row):
    """Extract available forms from target study row"""
    forms = set()
    form_fields = ['Primary Form Name', 'Form Name', 'CRF Name', 'Form']
    for field in form_fields:
        value = row.get(field, '')
        if value and str(value).strip() and str(value).strip().lower() not in ['n/a', 'nan', 'none']:
            # Handle comma-separated values
            form_list = [f.strip() for f in str(value).split(',') if f.strip()]
            forms.update(form_list)
    return list(forms)

def extract_target_visits(row):
    """Extract available visits from target study row"""
    visits = set()
    visit_fields = ['Primary Visit Name', 'Visit Name', 'Visit']
    for field in visit_fields:
        value = row.get(field, '')
        if value and str(value).strip() and str(value).strip().lower() not in ['n/a', 'nan', 'none']:
            # Handle comma-separated values
            visit_list = [v.strip() for v in str(value).split(',') if v.strip()]
            visits.update(visit_list)
    return list(visits)

def extract_target_variables(row):
    """Extract available variables from target study row"""
    variables = set()
    var_fields = ['Primary Domain Variables', 'Domain Variables', 'Variables', 'Primary Variables']
    for field in var_fields:
        value = row.get(field, '')
        if value and str(value).strip() and str(value).strip().lower() not in ['n/a', 'nan', 'none']:
            # Handle comma-separated values
            var_list = [v.strip() for v in str(value).split(',') if v.strip()]
            variables.update(var_list)
    return list(variables)

def convert_numpy_types(obj):
    """Recursively convert numpy types to regular Python types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

def process_mdd_file(filepath, upload_id=None, sponsor_name=None, match_scope='across'):
    """Process the uploaded MDD file"""
    try:
        logger.info(f"Processing MDD file: {filepath}")
        
        # Check if vector database is initialized
        if not enhanced_matching_engine.is_initialized():
            logger.info("Vector database not initialized, trying to load pre-computed embeddings...")
            
            # Try to load pre-computed embeddings first
            if not enhanced_matching_engine.load_precomputed_embeddings():
                logger.info("No pre-computed embeddings found, initializing from files...")
                database_files = []
                database_dir = 'MDD_DATABASE'
                
                for filename in os.listdir(database_dir):
                    if filename.lower().endswith(('.xlsx', '.csv')):
                        db_filepath = os.path.join(database_dir, filename)
                        database_files.append(db_filepath)
                
                if database_files:
                    enhanced_matching_engine.initialize_vector_db(database_files)
                else:
                    logger.warning("No database files found, proceeding without reference data")
        
        # Parse target MDD file
        target_data = file_processor.parse_target_mdd(filepath)
        # Persist raw Target MDD rows by upload_id for later lookup during script generation
        try:
            if upload_id:
                # Build a map dq_name_lower -> row
                _per_dq = {}
                for _row in (target_data or []):
                    _dq = _gf(_row, 'dq_name')
                    if _dq:
                        _per_dq[str(_dq).strip().lower()] = _row
                if _per_dq:
                    TARGET_MDD_ROWS[upload_id] = _per_dq
                    logger.info(f"Stored {len(_per_dq)} target rows for upload_id={upload_id}")
        except Exception as _store_ex:
            logger.warning(f"Failed to store Target MDD rows for upload_id={upload_id}: {_store_ex}")
        # Add sponsor_name to each row if provided
        if sponsor_name:
            for row in target_data:
                row['sponsor_name'] = sponsor_name
        ###
        if not target_data:
            return {'error': 'Failed to parse target MDD file or file is empty'}
        
        logger.info(f"Parsed {len(target_data)} rows from target file")
        
        # Process smaller subset for testing to avoid timeout
        results = []
        total_rows = len(target_data)
        
        # For large files, process only first 20 rows to demonstrate functionality
        ##RS - Commented out for now
        if total_rows > 50:
            target_data = target_data#[:10]
            logger.info(f"Large file detected ({total_rows} rows). Processing first {len(target_data)} rows for demonstration.")
            total_rows = len(target_data)
        
        logger.info(f"Processing {total_rows} rows")
        check_logic_mdd_df = load_aggregated_check_logic()
        for idx, row in enumerate(target_data):
            try:
                # Update progress periodically with base offset and monotonicity
                # Reserve 0–20% for upload/init, use 20–99% for processing rows
                stride = 1 if total_rows <= 50 else (5 if total_rows <= 300 else 20)
                if (idx % stride == 0) or (idx == total_rows - 1):
                    processed_pct = int(((idx + 1) / total_rows) * 80)  # allocate 80% range for processing
                    percent = 20 + processed_pct
                    if percent >= 100:
                        percent = 99  # leave 100% to finalization in upload_file
                    if upload_id is not None:
                        # Ensure monotonic progress (never go backwards)
                        prev = get_progress(upload_id)
                        prev_pct = int(prev.get('percent', 0)) if isinstance(prev, dict) else 0
                        if percent < prev_pct:
                            percent = prev_pct
                        set_progress(upload_id, percent, f"Processing row {idx + 1}/{total_rows}")
                logger.info(f"Processing row {idx + 1}/{total_rows}")
                
                # Use enhanced semantic matching with 4 improvements
                logger.info(f"Rs-line618 enhanced_matching_engine.py - logging row: {row}")
                # desired_top_k = 10 if (match_scope == 'same' and sponsor_name) else 3
                desired_top_k = 10 if (match_scope == 'same' and sponsor_name) else 3
                match_results = enhanced_matching_engine.find_matches(
                    row,
                    top_k=desired_top_k,
                    sponsor_filter=(sponsor_name if (match_scope == 'same' and sponsor_name) else None),
                    scope=('same' if (match_scope == 'same' and sponsor_name) else 'across')
                )
                
                # Take best match if available
                # logger.info(f"Rs-line621 enhanced_matching_engine.py - logging match_results: {match_results}")
                # if match_results:
                #     match_result = match_results[0]  # Best match
                #     logger.info(f'Rs-line624 enhanced_matching_engine.py - logging Final match_result: {match_result}')
                # else:
                #     logger.info(f'Rs-line626 enhanced_matching_engine.py - logging Final match_result: {match_result}') 
                #     match_result = None
                if  match_results:
                    match_result = match_results#[0]  # Best match
                    logger.info(f'Rs-line624 enhanced_matching_engine.py - logging Final match_result: {match_result}')
                else:
                    logger.info(f'Rs-line626 enhanced_matching_engine.py - logging Final match_result: {match_result}') 
                    match_result = None
                
                # Skip rows with no matches (returns None or empty list)
                if match_result is None:
                    logger.info(f"Row {idx + 1} filtered out - no matches found")
                    continue
                
                # Create enriched row with match data
                enriched_row = row.copy()
                # reference_data = match_result.get('reference_data', {})
                
                # Convert numpy types to regular Python types for JSON serialization
                def convert_numeric(value):
                    """Convert numpy types to regular Python types"""
                    if hasattr(value, 'item'):  # numpy scalar
                        return value.item()
                    elif isinstance(value, (int, float)):
                        return float(value)
                    return value
                
                # Extract reference data with proper field mapping
                # local get_field_value calls replaced with canonical _gf helper
                # logger.info(f"Rs-Line585 Field options: {field_options},{data}")
                # """Get first non-empty value from field options"""
                # for field in field_options:
                #     value = data.get(field, '')
                #     if value and str(value).strip() != '' and str(value).strip().lower() != 'n/a':
                #         return str(value)
                # return ''
                    #         return str(value)
                    # return ''
                
                # Add essential MDD columns with proper field names for frontend
                enriched_row.update({
                    'Is Match Found': match_result.get('is_match_found', 'NO'),
                    'Confidence Score': convert_numeric(match_result.get('confidence_score', 0.0)),
                    'Match Type': match_result.get('match_classification', 'No Match'),
                    'Match Reason': match_result.get('match_reason', ''),
                    'Match Explanation': match_result.get('match_explanation', 'No explanation available'),
                    'match_explanation': match_result.get('match_explanation', 'No explanation available'),  # Alternative field name
                    'Reference Check Name': _gf(match_result, 'dq_name'),
                    'Reference Check Description': _gf(match_result, 'dq_description'),
                    'Reference Query Text': _gf(match_result, 'query_text'),
                    'Reference Pseudo Code': _gf(match_result, 'pseudo_code'),
                    'Reference Sponsor': _gf(match_result, 'sponsor_name'),
                    'Reference Study': _gf(match_result, 'study_id'),
                    'Origin Study': (os.path.splitext(match_result.get('source_file', ''))[0].replace('_', ' ') if match_result.get('source_file') else ''),
                    # Enhanced metadata from matched database record
                    'P_Domain': _gf(match_result, 'primary_dataset'),
                    'P_form_name': _gf(match_result, 'P_form_name'),
                    'P_visit_name': _gf(match_result, 'P_visit_name'),
                    'Primary Domain Variables (Pre-Conf)': _gf(match_result, 'primary_dataset_columns'),
                    'Primary Dynamic Columns': _gf(match_result, 'dynamic_panel_variables'),
                    'R1_Domain': _gf(match_result, 'R1_Domain'),
                    'R2_Domain': _gf(match_result, 'R2_Domain'),
                    'R3_Domain': _gf(match_result, 'R3_Domain'),
                    'R4_Domain': _gf(match_result, 'R4_Domain'),
                    'R5_Domain': _gf(match_result, 'R5_Domain'),
                    'R1_form_name': _gf(match_result, 'R1_form_name'),
                    'R2_form_name': _gf(match_result, 'R2_form_name'),
                    'R3_form_name': _gf(match_result, 'R3_form_name'),
                    'R4_form_name': _gf(match_result, 'R4_form_name'),
                    'R5_form_name': _gf(match_result, 'R5_form_name'),
                    'R1_visit_name': _gf(match_result, 'R1_visit_name'),
                    'R2_visit_name': _gf(match_result, 'R2_visit_name'),
                    'R3_visit_name': _gf(match_result, 'R3_visit_name'),
                    'R4_visit_name': _gf(match_result, 'R4_visit_name'),
                    'R5_visit_name': _gf(match_result, 'R5_visit_name'),
                    'R1_Domain_Variables': _gf(match_result, 'R1_Domain_Variables'),
                    'R2_Domain_Variables': _gf(match_result, 'R2_Domain_Variables'),
                    'R3_Domain_Variables': _gf(match_result, 'R3_Domain_Variables'),
                    'R4_Domain_Variables': _gf(match_result, 'R4_Domain_Variables'),
                    'R5_Domain_Variables': _gf(match_result, 'R5_Domain_Variables'),
                    'R1_Dynamic_Columns': _gf(match_result, 'R1_Dynamic_Columns'),
                    'R2_Dynamic_Columns': _gf(match_result, 'R2_Dynamic_Columns'),
                    'R3_Dynamic_Columns': _gf(match_result, 'R3_Dynamic_Columns'),
                    'R4_Dynamic_Columns': _gf(match_result, 'R4_Dynamic_Columns'),
                    'R5_Dynamic_Columns': _gf(match_result, 'R5_Dynamic_Columns'),
                    'Check logic': _gf(match_result, 'pseudo_code') or lookup_aggregated_check_logic(check_logic_mdd_df,sponsor_name=_gf(match_result, 'sponsor_name'),study_id=_gf(match_result, 'study_id'),dq_name=_gf(match_result, 'dq_name')), 
                    ### TO BE CONTINUED
                    # 'Dynamic Panel Variables (Pre-Conf)': _gf(match_result, 'dynamic_panel_variables'),
                    'Query Target (Pre-Conf)': _gf(match_result, 'query_target'),
                    'Origin Study (Copy Source Study)': _gf(match_result, 'origin_study'),
                    'Pseudo Tech Code (Copy Source Study)': _gf(match_result, 'pseudo_tech_code'),
                    # Add Target fields from standardized columns (exact case match)
                    'Target Check Description': _gf(row, 'dq_description'),
                    'Target Query Text': (_gf(row, 'query_text') or _gf(row, 'pseudo_code'))
                })
                
                # Provide UI-expected keys and sensible fallbacks to avoid N/A in table
                # Map primary dataset/domain, form, visit to generic keys used by frontend
                enriched_row['Domain'] = _gf(match_result, 'primary_dataset') or enriched_row.get('P_Domain', '')
                enriched_row['Form Name'] = _gf(match_result, 'P_form_name') or enriched_row.get('P_form_name', '')
                enriched_row['Visit Name'] = _gf(match_result, 'P_visit_name') or enriched_row.get('P_visit_name', '')

                # Aggregate relational variables into a single field if not already present
                if not enriched_row.get('Relational Domain Variables'):
                    # logger.info(f'line762')
                    _rel_parts = []
                    for _k in ('R1_Domain_Variables', 'R2_Domain_Variables', 'R3_Domain_Variables', 'R4_Domain_Variables', 'R5_Domain_Variables'):
                        # _v = enriched_row.get(_k)
                        _v = _gf(match_result, _k)
                        if _v and str(_v).strip() and str(_v).strip().lower() not in ['n/a', 'na', 'nan', 'none', '']:
                            _rel_parts.append(str(_v).strip())
                    # logger.info(f'line768 {_rel_parts}')
                    if _rel_parts:
                        enriched_row['Relational Domain Variables'] = ', '.join(_rel_parts)

                # Aggregate relational domains (datasets) into a single field if not already present
                if not enriched_row.get('Relational Domains'):
                    _rel_dom_parts = []
                    for _k in ('R1_Domain', 'R2_Domain', 'R3_Domain', 'R4_Domain', 'R5_Domain'):
                        _v = _gf(match_result, _k)
                        if _v and str(_v).strip() and str(_v).strip().lower() not in ['n/a', 'na', 'nan', 'none', '']:
                            _rel_dom_parts.append(str(_v).strip())
                    if _rel_dom_parts:
                        enriched_row['Relational Domains'] = ', '.join(_rel_dom_parts)

                # Aggregate relational dynamic columns into a single field if not already present
                if not enriched_row.get('Relational Dynamic Variables'):
                    _rel_dyn_parts = []
                    for _k in ('R1_Dynamic_Columns', 'R2_Dynamic_Columns', 'R3_Dynamic_Columns', 'R4_Dynamic_Columns', 'R5_Dynamic_Columns'):
                        _v = _gf(match_result, _k)
                        if _v and str(_v).strip() and str(_v).strip().lower() not in ['n/a', 'na', 'nan', 'none', '']:
                            _rel_dyn_parts.append(str(_v).strip())
                    if _rel_dyn_parts:
                        enriched_row['Relational Dynamic Variables'] = ', '.join(_rel_dyn_parts)

                # Fallback for dynamic variables if canonical field is missing but primary dynamic columns exist
                if not enriched_row.get('Dynamic Panel Variables (Pre-Conf)'):
                    # logger.info(f'line773')
                    _dyn = _gf(match_result, 'dynamic_panel_variables') or _gf(enriched_row, 'dynamic_panel_variables')
                    # logger.info(f'line775 {_dyn}')
                    if _dyn and str(_dyn).strip():
                        enriched_row['Dynamic Panel Variables (Pre-Conf)'] = str(_dyn).strip()

                
                 
                # Add validation logic and operational notes
                operational_notes = []
                
                # Get target study metadata (Pre-Conf) - this would typically come from a separate configuration file 
                # For now, we'll extract available information from the target row
                # RS - Try to Fetch these information from SAM or PRE_Conformance atleast and Map Tables
                target_forms = extract_target_forms(row) 
                target_visits = extract_target_visits(row)
                target_variables = extract_target_variables(row)
                
                # # Validate form existence
                # matched_form = enriched_row.get('Form Name', '')
                # if matched_form and matched_form not in target_forms and target_forms:
                #     operational_notes.append(f"FORM REMOVED: '{matched_form}' does not exist in target study")
                #     enriched_row['Form Name'] = ''  # Remove the form
                
                # # Validate visit existence
                # matched_visit = enriched_row.get('Visit Name', '')
                # if matched_visit and matched_visit not in target_visits and target_visits:
                #     operational_notes.append(f"VISIT REMOVED: '{matched_visit}' does not exist in target study")
                #     enriched_row['Visit Name'] = ''  # Remove the visit
                
                # Validate variable existence ##RS ->PENDING its now comparing against Input MDD primary domain variables, but ideally we should check from SAM or PRE_Conformance atleast
                # matched_variables = enriched_row.get('Primary Domain Variables (Pre-Conf)', '')
                # if matched_variables and target_variables:
                #     matched_var_list = [v.strip() for v in matched_variables.split(',') if v.strip()]
                #     valid_variables = []
                #     for var in matched_var_list:
                #         if var in target_variables:
                #             valid_variables.append(var)
                #         else:
                #             operational_notes.append(f"VARIABLE REMOVED: '{var}' does not exist in target study")
                #     enriched_row['Primary Domain Variables (Pre-Conf)'] = ', '.join(valid_variables)
                
                # Add operational notes to the row
                if operational_notes:
                    enriched_row['Operational Notes'] = ' | '.join(operational_notes)
                else:
                    enriched_row['Operational Notes'] = ''
                logger.info(f'Rs-line841 app.py - logging enriched_row: {enriched_row}')
                results.append(enriched_row)
                
                # Log progress
                if (idx + 1) % 5 == 0 or idx + 1 == total_rows:
                    logger.info(f"Completed {idx + 1}/{total_rows} rows ({((idx + 1) / total_rows * 100):.1f}%)")
                
            except Exception as e:
                logger.warning(f"Error processing row {idx + 1}: {str(e)}")
                # Add row with error note
                error_row = row.copy()
                error_row['Match Type'] = 'Error'
                error_row['Confidence Score'] = 0
                error_row['Operational Note'] = f'Processing error: {str(e)}'
                results.append(error_row)
        
        # Generate output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = os.path.splitext(os.path.basename(filepath))[0]
        
        # Generate enriched MDD CSV file (all results)
        csv_filename = f"{base_filename}_enriched_{timestamp}.csv"
        csv_path = os.path.join(app.config['OUTPUT_FOLDER'], csv_filename)
        logger.info(f"Generating enriched MDD CSV file: {results}")
        
        mdd_generator.generate_csv_output(results, csv_path)
        # If using S3, upload the generated CSV
        if USE_S3:
            try:
                with open(csv_path, 'rb') as _f:
                    from utils.storage import storage as _storage
                    _storage.write_bytes(f"{app.config['OUTPUT_FOLDER']}/{csv_filename}", _f.read())
            except Exception as _e:
                logger.error(f"Failed to upload CSV to storage: {str(_e)}")
        
        # Filter matches for the matches-only file (exclude "No Match" but include all other categories)
        matched_results = [r for r in results if r.get('Match Type') != 'No Match' and r.get('Match Type') != 'Error']
        # Sort matched results by confidence score (already sorted but maintain consistency)
        matched_results.sort(key=lambda x: float(x.get('Confidence Score', 0)), reverse=True)
        
        # Generate matches-only CSV file
        matches_csv_filename = f"{base_filename}_matches_only_{timestamp}.csv"
        matches_csv_path = os.path.join(app.config['OUTPUT_FOLDER'], matches_csv_filename)
        # if matched_results: #Not required since it only filters out No Match and Error
            # mdd_generator.generate_csv_output(matched_results, matches_csv_path)
        
        # Generate detailed JSON results (all results)
        json_filename = f"{base_filename}_results_{timestamp}.json"
        json_path = os.path.join(app.config['OUTPUT_FOLDER'], json_filename)
        
        # Generate matches-only JSON file
        matches_json_filename = f"{base_filename}_matches_only_{timestamp}.json"
        matches_json_path = os.path.join(app.config['OUTPUT_FOLDER'], matches_json_filename)
        
        # Sort results by confidence score in descending order (best matches first)
        results.sort(key=lambda x: float(x.get('Confidence Score', 0)), reverse=True)
        
        # Calculate statistics
        stats = calculate_statistics(results)
        
        detailed_results = {
            'summary': stats,
            'processed_at': datetime.now().isoformat(),
            'target_file': os.path.basename(filepath),
            'total_rows': len(results),
            # 'results': results
        }
        
        # Save all results JSON
        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        # If using S3, upload the generated JSON
        if USE_S3:
            try:
                with open(json_path, 'rb') as _f:
                    from utils.storage import storage as _storage
                    _storage.write_bytes(f"{app.config['OUTPUT_FOLDER']}/{json_filename}", _f.read())
            except Exception as _e:
                logger.error(f"Failed to upload JSON to storage: {str(_e)}")
        
        # Save matches-only JSON
        # if matched_results:
        #     matches_detailed_results = {
        #         'summary': {
        #             'total_rows': len(matched_results),
        #             'complete_matches': sum(1 for r in matched_results if r.get('Match Type') == 'Complete'),
        #             'partial_matches': sum(1 for r in matched_results if r.get('Match Type') == 'Partial'),
        #             'no_matches': 0
        #         },
        #         'processed_at': datetime.now().isoformat(),
        #         'target_file': os.path.basename(filepath),
        #         'filter': 'Matches and Partial Matches Only',
        #         'total_rows': len(matched_results),
        #         'results': matched_results
        #     }
            
        #     with open(matches_json_path, 'w') as f:
        #         json.dump(matches_detailed_results, f, indent=2, default=str)
        
        logger.info(f"Processing completed. Generated {csv_filename}, {json_filename}, {matches_csv_filename}, and {matches_json_filename}")
        
        # Convert all data to ensure JSON serialization compatibility
        response_data = {
            'success': True,
            'message': 'MDD processing completed successfully',
            'statistics': stats,
            'results': matched_results,  # Only show high-quality matches in frontend table
            'files': {
                'csv': csv_filename,
                'json': json_filename,
                # 'matches_csv': matches_csv_filename if matched_results else None,
                # 'matches_json': matches_json_filename if matched_results else None
            },
            'download_urls': {
                'csv': url_for('download_file', filename=csv_filename),
                'json': url_for('download_file', filename=json_filename),
                # 'matches_csv': url_for('download_file', filename=matches_csv_filename) if matched_results else None,
                # 'matches_json': url_for('download_file', filename=matches_json_filename) if matched_results else None
            }
        }
        
        # Apply type conversion to entire response to fix JSON serialization
        return convert_numpy_types(response_data)
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        return {'error': f'Processing failed: {str(e)}'}

def calculate_statistics(results):
    """Calculate processing statistics with new color-coded classifications"""
    total = len(results)
    
    # Green category (good matches)
    excellent_matches = sum(1 for r in results if r.get('Match Type') in ('Excellent Match', 'Excellent'))
    good_matches = sum(1 for r in results if r.get('Match Type') in ('Good Match', 'Good'))
    
    # Amber category (moderate/weak matches)
    moderate_matches = sum(1 for r in results if r.get('Match Type') in ('Moderate Match', 'Moderate'))
    # Treat legacy 'Low Match' and new 'Weak Match' as the same bucket
    low_matches = sum(1 for r in results if r.get('Match Type') in ('Low Match', 'Weak Match', 'Weak'))
    
    # Red category (no matches)
    no_matches = sum(1 for r in results if r.get('Match Type') == 'No Match')
    errors = sum(1 for r in results if r.get('Match Type') == 'Error')
    
    # Legacy support for old classifications
    legacy_complete = sum(1 for r in results if r.get('Match Type') == 'Complete')
    legacy_partial = sum(1 for r in results if r.get('Match Type') == 'Partial')
    
    # Combined categories
    total_green = excellent_matches + good_matches + legacy_complete
    total_amber = moderate_matches + low_matches + legacy_partial
    total_matches = total_green + total_amber
    
    return {
        'total_rows': total,
        'complete_matches': total_green,  # Green category for UI compatibility
        'partial_matches': total_amber,   # Amber category for UI compatibility
        'no_matches': no_matches,
        'errors': errors,
        'match_rate': round(total_matches / total * 100, 1) if total > 0 else 0,
        # Detailed breakdown
        'excellent_matches': excellent_matches,
        'good_matches': good_matches,
        'moderate_matches': moderate_matches,
        # 'low_matches': low_matches,
        # Provide weak_matches as an alias for UI robustness
        'weak_matches': low_matches
    }
def load_aggregated_check_logic():
    # load aggregated check logic from database
    try:
        # Load CSV as DataFrame from S3 or local, cache between calls with a signature
        if USE_S3:
            key = 'Check_logic_collated_MDD/Output_ref_mdd/aggregated_checks.csv'
            data = storage.read_bytes(key)
            sig = ('s3', len(data))
            if getattr(load_aggregated_check_logic, '_cache_sig', None) != sig:
                df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False)
                df.columns = [str(c).strip().lower() for c in df.columns]
                load_aggregated_check_logic._df_cache = df
                load_aggregated_check_logic._cache_sig = sig
            else:
                df = getattr(load_aggregated_check_logic, '_df_cache', None)
        else:
            csv_path = os.path.join(
                os.getcwd(),
                'Check_logic_collated_MDD',
                'Output_ref_mdd',
                'aggregated_checks.csv'
            )
            if not os.path.exists(csv_path):
                logger.warning('Aggregated CSV not found at %s', csv_path)
                return pd.DataFrame()
            mtime = os.path.getmtime(csv_path)
            sig = ('local', mtime)
            if getattr(load_aggregated_check_logic, '_cache_sig', None) != sig:
                df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
                df.columns = [str(c).strip().lower() for c in df.columns]
                load_aggregated_check_logic._df_cache = df
                load_aggregated_check_logic._cache_sig = sig
            else:
                df = getattr(load_aggregated_check_logic, '_df_cache', None)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"load_aggregated_check_logic error: {e}")
        return pd.DataFrame()

def lookup_aggregated_check_logic(df,sponsor_name,study_id,dq_name):
    """Match a row in aggregated_checks DataFrame using simple token-based LIKE across
    sponsor_name, study_id, dq_name and return the 'check_logic' string.
    """
    try:
        
        if df is None or df.empty:
            return 'N/A'
        logger.info(f"Rs-line1182 lookup_aggregated_check_logic - sponsor_name: {sponsor_name}, study_id: {study_id}, dq_name: {dq_name}, df: {len(df)}")
        sponsor_name = str(sponsor_name).strip().lower()
        study_id = str(study_id).strip().lower()
        dq_name = str(dq_name).strip().lower()
        # dq_name_search_words = set(re.split(r'[\s_-]+', dq_name.lower()))

        df = df[((df['sponsor_name'].str.lower().str.strip().str.contains(sponsor_name)) | (df['sponsor_name'].str.lower().str.strip().apply(lambda x: x in sponsor_name))) 
                & ((df['study_id'].str.lower().str.strip().str.contains(study_id)) | (df['study_id'].str.lower().str.strip().apply(lambda x: x in study_id))) 
                & ((df['dq_name'].str.lower().str.strip().str.contains(dq_name)) | (df['dq_name'].str.lower().str.strip().apply(lambda x: x in dq_name)) | (df['dq_name'].str.lower().str.strip().str.replace(' ', '_').str.replace('-','_').apply(lambda x: x in dq_name)))
                # & (df['dq_name'].str.lower().str.strip().apply(lambda s: any(word in dq_name_search_words for word in re.split(r'[\s_-]+', s.lower()))))
                ]
        check_logic = df['check_logic'].values[0] if not df.empty else 'N/A'
        logger.info(f"Rs-line1189 lookup_aggregated_check_logic : check_logic: {check_logic}")
        return check_logic
    except Exception as e:
        logger.error(f"lookup_aggregated_check_logic error: {e}")
        return 'N/A'
    

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size is {MAX_UPLOAD_MB}MB.'}), 413

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=False)
