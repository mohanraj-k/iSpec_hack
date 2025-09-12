# Auto MDD System Architecture

## Overview

Auto MDD is a Flask-based web application that automates Master Data Definition (MDD) template creation through AI-powered semantic matching. The system uses OpenAI embeddings with numpy-based vector search to match target validation rules against a comprehensive library of DQ specifications.

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                          Frontend Layer                         │
├─────────────────────────────────────────────────────────────────┤
│  HTML Templates (Jinja2) + Bootstrap CSS + Vanilla JavaScript   │
│  ├── File Upload Interface                                      │
│  ├── Progress Tracking                                          │
│  ├── Results Table with Quality Filtering                       │
│  └── Database Management Controls                               │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
├─────────────────────────────────────────────────────────────────┤
│                      Flask Web Server                          │
│  ├── REST API Endpoints (/upload, /rebuild-database, etc.)     │
│  ├── Request/Response Handling                                 │
│  ├── File Upload Management                                    │
│  └── Session Management                                        │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Service Layer                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ File Processing │ │ Matching Engine │ │ Output Gen.     │   │
│  │ - Excel parsing │ │ - Vector search │ │ - CSV/JSON      │   │
│  │ - Field mapping │ │ - Hybrid scoring│ │ - Enrichment    │   │
│  │ - Validation    │ │ - Classification│ │ - Download mgmt │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                         AI/ML Layer                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ OpenAI Client   │ │ Vector Database │ │ Text Processing │   │
│  │ - Embeddings    │ │ - Numpy arrays  │ │ - Cleaning      │   │
│  │ - API handling  │ │ - Cosine sim.   │ │ - Enhancement   │   │
│  │ - Retry logic   │ │ - Indexing      │ │ - Synonyms      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐   │
│  │ MDD_DATABASE/   │ │ data/           │ │ output/         │   │
│  │ - Reference MDDs│ │ - Embeddings    │ │ - Results       │   │
│  │ - Excel files   │ │ - Metadata      │ │ - Downloads     │   │
│  │ - 722 records   │ │ - Vector index  │ │ - CSV/JSON      │   │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Flask Application (`app.py`)

**Purpose**: Main application entry point and request router

**Key Functions**:
- `index()`: Serves main application page
- `upload_file()`: Handles file upload and processing orchestration
- `initialize_database()`: Manages vector database initialization
- `rebuild_database()`: Triggers dynamic database rebuilding
- `database_summary()`: Provides database statistics
- `download_file()`: Manages result file downloads

**Architecture Patterns**:
- **MVC Pattern**: Controllers handle requests, models process data, views render responses
- **Service Layer**: Delegates business logic to specialized utility classes
- **Error Handling**: Comprehensive exception handling with user-friendly messages
- **File Management**: Secure file upload with validation and cleanup

**Key Design Decisions**:
```python
# Secure file handling
if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    
# Async-like processing with progress tracking
def process_mdd_file(filepath):
    # Progress tracking for long-running operations
    for i, row in enumerate(target_data):
        logger.info(f"Completed {i+1}/{len(target_data)} rows ({((i+1)/len(target_data)*100):.1f}%)")
```

### 2. Numpy Matching Engine (`utils/numpy_matching_engine.py`)

**Purpose**: Core semantic matching and vector search engine

**Architecture Overview**:
```
Target Text → Text Enhancement → OpenAI Embedding → Vector Search → Hybrid Scoring → Classification
```

**Key Classes and Methods**:

**`NumpyMatchingEngine`**:
- **State Management**: Maintains embeddings array and metadata dictionary
- **Initialization**: `load_precomputed_embeddings()` - loads saved vector database
- **Search**: `find_matches()` - performs semantic similarity search
- **Database Management**: `rebuild_vector_database()` - regenerates embeddings

**Core Algorithm Flow**:
```python
def find_matches(self, target_row: Dict[str, Any], top_k: int = 5):
    # 1. Text Enhancement
    embedding_text = self._create_enhanced_embedding_text(target_row)
    
    # 2. OpenAI Embedding Generation
    target_embedding = self.client.get_embedding(embedding_text)
    
    # 3. Vector Search (Cosine Similarity)
    similarities = np.dot(self.embeddings, target_embedding)
    
    # 4. Hybrid Scoring
    for idx in top_indices:
        cosine_sim = similarities[idx]
        keyword_overlap = self._calculate_keyword_overlap(target_row, ref_data)
        final_score = 0.7 * cosine_sim + 0.3 * keyword_overlap
    
    # 5. Classification and Ranking
    classification = self._classify_match_enhanced(final_score)
```

**Text Enhancement Pipeline**:
```python
def _create_enhanced_embedding_text(self, row):
    # Extract key components
    check_name = self._extract_field_value(row, ['DQ Name', 'Check Name'])
    description = self._extract_field_value(row, ['DQ Description', 'EC Description'])
    query_text = self._extract_field_value(row, ['Standard Query text', 'Query Text'])
    
    # Remove generic phrases
    description = self._remove_generic_phrases(description)
    
    # Structure for embedding
    components = []
    if check_name: components.append(f"CHECK:{check_name}")
    if description: components.append(f"DESC:{description}")
    if query_text: components.append(f"QUERY:{query_text}")
    
    return " ".join(components)
```

**Hybrid Scoring Model**:
- **70% Semantic Similarity**: OpenAI embedding cosine similarity
- **30% Keyword Overlap**: Jaccard similarity with clinical term expansion
- **Reuse Penalty**: 5% reduction per reference reuse (max 15%)

**Classification Thresholds**:
```python
def _classify_match_enhanced(self, score: float):
    if score >= 0.35: return "Excellent Match"
    elif score >= 0.25: return "Good Match"  
    elif score >= 0.15: return "Moderate Match"
    elif score >= 0.05: return "Weak Match"
    else: return "No Match"
```

### 3. OpenAI Client (`utils/azure_openai_client.py`)

**Purpose**: Manages OpenAI API interactions and embedding generation

**Key Features**:
- **API Abstraction**: Wraps OpenAI client with retry logic and error handling
- **Batch Processing**: Efficient handling of multiple embedding requests
- **Text Cleaning**: Preprocessing for optimal embedding quality
- **Rate Limiting**: Handles API constraints gracefully

**Core Methods**:
```python
def get_embedding(self, text: str, max_retries: int = 3):
    # Clean and validate input
    cleaned_text = self._clean_text(text)
    
    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = self.client.embeddings.create(
                input=cleaned_text,
                model="text-embedding-3-small"
            )
            return response.data[0].embedding
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e
```

**Text Cleaning Pipeline**:
```python
def clean_logic_text(self, text: str):
    # Remove verbose prefixes
    text = re.sub(r'^(query should fire|dm review|please update)[:,-]?\s*', '', text, flags=re.IGNORECASE)
    
    # Remove generic phrases
    generic_phrases = ['when applicable', 'if necessary', 'as needed']
    for phrase in generic_phrases:
        text = text.replace(phrase, '')
    
    return text.strip()
```

### 4. File Processor (`utils/mdd_file_processor.py`)

**Purpose**: Excel file parsing with flexible column mapping

**Design Pattern**: **Strategy Pattern** for different MDD file formats

**Key Features**:
- **Flexible Field Mapping**: Handles various column name variations
- **Robust Parsing**: Uses openpyxl for reliable Excel processing
- **Validation**: Ensures data quality and completeness
- **Error Recovery**: Graceful handling of malformed files

**Column Mapping Strategy**:
```python
def _extract_field_value(self, row_dict, field_options):
    """Extract first non-empty value from field options"""
    for field in field_options:
        # Case-insensitive matching
        for key, value in row_dict.items():
            if key and field.lower() in key.lower():
                if value and str(value).strip():
                    return str(value).strip()
    return ""

# Flexible field mappings
TARGET_FIELD_MAPPINGS = {
    'dq_name': ['DQ Name', 'Check Name', 'Check ID'],
    'dq_description': ['DQ Description', 'EC Description', 'Check Description'],
    'standard_query': ['Standard Query text', 'Query Text', 'Query'],
    'form_name': ['Primary Form Name', 'Form Name', 'Domain'],
    'visit_name': ['Primary Visit Name', 'Visit Name', 'Visit']
}
```

**File Processing Workflow**:
```python
def parse_target_mdd(self, file_path: str):
    # Load workbook
    workbook = openpyxl.load_workbook(file_path, data_only=True)
    worksheet = workbook.active
    
    # Extract headers
    headers = [cell.value for cell in worksheet[1]]
    
    # Process data rows
    results = []
    for row in worksheet.iter_rows(min_row=2, values_only=True):
        row_dict = dict(zip(headers, row))
        if self._is_valid_target_row(row_dict):
            results.append(row_dict)
    
    return results
```

### 5. Output Generator (`utils/mdd_output_generator.py`)

**Purpose**: Creates enriched output files with comprehensive metadata

**Output Formats**:
- **CSV**: Tabular data for Excel compatibility
- **JSON**: Structured data with full metadata
- **Matches-only**: Filtered results containing only successful matches

**Key Features**:
- **Metadata Enrichment**: Adds 23 comprehensive fields
- **Excel Formatting**: Professional styling with color coding
- **Data Validation**: Ensures output integrity
- **Download Management**: Secure file serving

**Output Schema**:
```python
COMPREHENSIVE_OUTPUT_COLUMNS = [
    # Target data
    'Target DQ Name', 'Target Check Description', 'Target Query Text',
    'Target Form Name', 'Target Visit Name',
    
    # Match metadata
    'Is Match Found', 'Confidence Score', 'Match Type', 'Match Reason',
    'Match Explanation',
    
    # Reference data
    'Reference Check Name', 'Reference Check Description', 
    'Reference Query Text', 'Reference Form Name', 'Reference Visit Name',
    
    # Extracted metadata (green headers)
    'Domain', 'Extracted Form Names', 'Extracted Visit Names',
    'Primary Variables', 'Relational Variables', 'Dynamic Panel Variables',
    'Query Target', 'Origin Study', 'Pseudo Tech Code',
    
    # Operational notes (red header)
    'Operational Notes'
]
```

**Excel Enhancement**:
```python
def generate_excel_output(self, results, output_path):
    # Create workbook with styling
    workbook = Workbook()
    worksheet = workbook.active
    
    # Apply color-coded headers
    for col_num, header in enumerate(headers, 1):
        cell = worksheet.cell(row=1, column=col_num, value=header)
        
        if header in METADATA_COLUMNS:
            cell.fill = PatternFill(start_color="90EE90", end_color="90EE90", fill_type="solid")  # Light green
        elif header == 'Operational Notes':
            cell.fill = PatternFill(start_color="FFB6C1", end_color="FFB6C1", fill_type="solid")  # Light red
        
        cell.font = Font(bold=True)
```

### 6. Frontend Architecture

**Technology Stack**:
- **HTML Templates**: Jinja2 templating with Bootstrap 5
- **CSS Framework**: Bootstrap with custom dark theme
- **JavaScript**: Vanilla JS with modern ES6+ features
- **Icons**: Font Awesome for UI elements

**Key JavaScript Components**:

**`MDDAutomationApp` Class**:
```javascript
class MDDAutomationApp {
    constructor() {
        this.currentResults = [];
        this.init();
    }
    
    // Core functionality
    async handleFileUpload() { /* File upload with progress tracking */ }
    filterResults(filterType) { /* Table filtering logic */ }
    filterByQuality(qualityLevel) { /* Quality slider filtering */ }
    updateStatistics(filteredResults) { /* Dynamic statistics updates */ }
}
```

**Quality Filter Implementation**:
```javascript
applyQualityFilter(results, qualityLevel) {
    const qualityFilters = {
        1: ['Excellent Match'],                                           // Excellent only
        2: ['Excellent Match', 'Good Match'],                            // Good & above
        3: ['Excellent Match', 'Good Match', 'Moderate Match'],          // Moderate & above  
        4: ['Excellent Match', 'Good Match', 'Moderate Match', 'Weak Match'] // All matches
    };
    
    const allowedTypes = qualityFilters[qualityLevel] || qualityFilters[1];
    return results.filter(result => {
        const matchType = result['Match Type'] || '';
        return allowedTypes.includes(matchType);
    });
}
```

**Progressive Enhancement Pattern**:
- Base functionality works without JavaScript
- Enhanced features added through JavaScript
- Graceful degradation for accessibility

## Data Storage Architecture

### No Traditional Database Design

The Auto MDD system uses a **file-based storage architecture** without traditional SQL or NoSQL databases. This design choice optimizes for the specific use case of semantic similarity search and provides several advantages.

### Storage Components

**1. Vector Database (File-Based)**
```
data/
├── precomputed_embeddings.npy    # 722 x 1536 numpy array of OpenAI embeddings
├── precomputed_metadata.json     # Corresponding metadata for each embedding
└── README.txt                    # Database information and build timestamp
```

**Implementation Details**:
```python
# Loading vector database
self.embeddings = np.load('data/precomputed_embeddings.npy')
with open('data/precomputed_metadata.json', 'r') as f:
    self.metadata = json.load(f)

# Vector similarity search (no database queries)
similarities = np.dot(self.embeddings, target_embedding)
top_indices = np.argsort(similarities)[::-1][:top_k]
```

**2. Reference Data Storage**
```
MDD_DATABASE/
├── File1_MDD.xlsx              # Source Excel files
├── File2_MDD.xlsx              # 722 total DQ specifications
└── FileN_MDD.xlsx              # Direct file system access
```

**3. Temporary Storage**
```
uploads/                        # Temporary upload storage
├── secure_filename.xlsx        # User uploads (auto-cleanup)
└── processing_temp/            # Processing workspace

output/                         # Generated results
├── enriched_YYYYMMDD_HHMMSS.csv
├── results_YYYYMMDD_HHMMSS.json
└── matches_only_YYYYMMDD_HHMMSS.json
```

**4. Application State**
```python
# In-memory storage only
class NumpyMatchingEngine:
    def __init__(self):
        self.embeddings = None          # Loaded into memory
        self.metadata = None            # Loaded into memory
        self.is_initialized = False     # Runtime state

# No persistent sessions or user data
app.config['SESSION_TYPE'] = 'filesystem'  # Temporary only
```

### Why No Traditional Database?

**Performance Advantages**:
- **Millisecond Search**: Numpy vectorized operations faster than database queries
- **No Query Overhead**: Direct memory access to embedding vectors
- **Batch Operations**: Efficient similarity calculations across all 722 vectors
- **Cache Efficiency**: Data loaded once, used repeatedly

**Operational Simplicity**:
- **Zero Configuration**: No database setup, connection pools, or schemas
- **Self-Contained**: Entire system runs from file system
- **Easy Backup**: Simple file copy for data backup
- **No Dependencies**: Eliminates database server requirements

**Data Access Patterns**:
- **Read-Heavy**: 99% read operations (similarity search)
- **Infrequent Writes**: Only during database rebuild
- **No Transactions**: No complex data relationships
- **Stateless**: Each request independent

### Performance Characteristics

**Vector Search Performance**:
```python
# Benchmark: 722 vectors × 1536 dimensions
search_time = measure_time(lambda: np.dot(embeddings, target_vector))
# Result: < 1ms for complete similarity search

# Database equivalent would require:
# 1. Query to fetch all embeddings
# 2. Transfer 722 × 1536 floats over network
# 3. Compute similarities
# 4. Sort and return results
# Total: 50-200ms typical database query time
```

**Memory Usage**:
```
Embeddings Array: 722 × 1536 × 4 bytes = ~4.4 MB
Metadata JSON: ~2.0 MB (text fields)
Total Memory: ~6.4 MB for entire database
```

### Data Persistence Strategy

**Precomputed Embeddings**:
- Generated once during database build
- Saved as binary numpy arrays for fast loading
- Metadata stored as JSON for flexibility

**Rebuild Process**:
```python
def rebuild_vector_database():
    # Process all MDD files
    all_records = []
    all_embeddings = []
    
    for mdd_file in glob.glob('MDD_DATABASE/*.xlsx'):
        records = parse_excel_file(mdd_file)
        for record in records:
            embedding = openai_client.get_embedding(record)
            all_records.append(record)
            all_embeddings.append(embedding)
    
    # Save to disk
    np.save('data/precomputed_embeddings.npy', np.array(all_embeddings))
    with open('data/precomputed_metadata.json', 'w') as f:
        json.dump(all_records, f)
```

**File System Organization**:
```
Auto-MDD/
├── data/                       # Vector database
│   ├── precomputed_embeddings.npy
│   └── precomputed_metadata.json
├── MDD_DATABASE/               # Source reference files
│   └── *.xlsx
├── uploads/                    # Temporary user files
├── output/                     # Generated results
└── static/                     # Application assets
```

### Comparison: File-Based vs Database

| Aspect | File-Based (Current) | Database Alternative |
|--------|---------------------|---------------------|
| Setup Complexity | None | PostgreSQL + vector extension |
| Search Speed | <1ms | 10-50ms |
| Memory Usage | 6.4MB | + DB overhead |
| Deployment | Single binary | + Database server |
| Backup | File copy | DB dump/restore |
| Scaling | Vertical only | Horizontal possible |
| ACID | Not needed | Available |
| Concurrent Users | Limited | Unlimited |

### Future Database Considerations

**When Traditional Database Would Be Beneficial**:

**User Management Requirements**:
```sql
-- Would require if adding user accounts
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100),
    preferences JSONB
);

CREATE TABLE processing_history (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    filename VARCHAR(255),
    processed_at TIMESTAMP,
    results JSONB
);
```

**Analytics and Monitoring**:
```sql
-- For usage tracking and optimization
CREATE TABLE search_analytics (
    id SERIAL PRIMARY KEY,
    query_text TEXT,
    results_count INT,
    avg_confidence FLOAT,
    processing_time_ms INT,
    created_at TIMESTAMP
);
```

**Distributed Architecture**:
- **Vector Database**: Pinecone, Weaviate, or Qdrant for distributed vector search
- **Metadata Store**: PostgreSQL for structured data
- **Cache Layer**: Redis for frequently accessed results

### Data Integrity and Reliability

**Current Approach**:
- **Atomic Updates**: Replace entire files during rebuild
- **Validation**: Verify embedding count matches metadata count
- **Error Handling**: Graceful fallback if files corrupted

**File System Reliability**:
```python
def load_precomputed_embeddings():
    try:
        # Validate file integrity
        embeddings = np.load(embeddings_path)
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Verify consistency
        if len(embeddings) != len(metadata):
            raise ValueError("Embedding/metadata count mismatch")
            
        return embeddings, metadata
    except Exception as e:
        logger.error(f"Database load failed: {e}")
        return None, None
```

This file-based approach provides optimal performance for the current semantic matching use case while maintaining simplicity and reliability.

## Data Flow Architecture

### 1. File Upload and Processing Flow

```
User Upload → File Validation → Excel Parsing → Field Extraction → Text Enhancement → Embedding Generation → Vector Search → Result Compilation → Output Generation
```

**Detailed Steps**:
1. **File Upload**: Secure file handling with size and format validation
2. **Excel Parsing**: openpyxl-based parsing with flexible column mapping
3. **Field Extraction**: Intelligent field matching across different MDD formats
4. **Text Enhancement**: Cleaning and structuring for optimal embeddings
5. **Embedding Generation**: OpenAI API calls with retry logic
6. **Vector Search**: Numpy-based cosine similarity search
7. **Hybrid Scoring**: Combined semantic and keyword-based scoring
8. **Classification**: Confidence-based match categorization
9. **Output Generation**: CSV and JSON file creation with metadata
10. **Download Management**: Secure file serving with cleanup

### 2. Database Rebuild Flow

```
User Trigger → File Discovery → Excel Processing → Embedding Generation → Vector Index Creation → Metadata Storage → UI Update
```

**Implementation**:
```python
def rebuild_vector_database(self, mdd_files: List[str]):
    all_data = []
    
    # Process each MDD file
    for file_path in mdd_files:
        file_data = self.file_processor.parse_reference_mdd(file_path)
        all_data.extend(file_data)
    
    # Generate embeddings
    embeddings = []
    for record in all_data:
        embedding_text = self._create_enhanced_embedding_text(record)
        embedding = self.client.get_embedding(embedding_text)
        embeddings.append(embedding)
    
    # Save to disk
    self.embeddings = np.array(embeddings)
    self.metadata = all_data
    self._save_precomputed_embeddings()
```

## Performance Optimizations

### 1. Vector Search Optimization

**Precomputed Embeddings**: All reference embeddings computed once and cached
**Numpy Vectorization**: Efficient batch operations for similarity calculations
**Memory Management**: Lazy loading and efficient data structures

### 2. API Optimization

**Batch Processing**: Group multiple embedding requests when possible
**Retry Logic**: Exponential backoff for API failures
**Rate Limiting**: Respect OpenAI API constraints

### 3. Frontend Optimization

**Progressive Loading**: Load results incrementally for large datasets
**Efficient Filtering**: Client-side filtering without server round-trips
**Memory Management**: Cleanup of large result sets

## Security Architecture

### 1. File Upload Security

```python
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'xlsx'}

def secure_upload(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Additional validation and sanitization
```

### 2. API Security

**Input Validation**: Comprehensive validation of all inputs
**Error Handling**: Secure error messages without information leakage
**Session Management**: Secure session handling with proper secret keys

### 3. Data Privacy

**Temporary Storage**: Automatic cleanup of uploaded files
**No Persistent Storage**: User data not permanently stored
**API Key Security**: Secure handling of OpenAI credentials

## Monitoring and Logging

### 1. Application Logging

```python
import logging

# Structured logging with levels
logger = logging.getLogger(__name__)
logger.info(f"Processing file: {filename}")
logger.error(f"Error in processing: {str(e)}")
```

### 2. Performance Monitoring

**Processing Time Tracking**: Monitor embedding generation and search times
**API Usage Monitoring**: Track OpenAI API usage and costs
**Error Rate Monitoring**: Track and alert on processing failures

### 3. User Activity Tracking

**Upload Statistics**: Track file sizes and processing times
**Feature Usage**: Monitor quality filter and download patterns
**Error Analysis**: Analyze common failure patterns

## Deployment Architecture

### 1. Production Configuration

```python
# Production settings
app.config.update(
    SECRET_KEY=os.environ.get("SESSION_SECRET"),
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB limit
    UPLOAD_FOLDER='uploads',
    OUTPUT_FOLDER='output'
)
```

### 2. Environment Management

**Environment Variables**: Secure configuration management
**Docker Ready**: Containerization support for deployment
**Health Checks**: Application health monitoring endpoints

### 3. Scalability Considerations

**Stateless Design**: No server-side session dependencies
**Horizontal Scaling**: Support for load balancing
**Database Scaling**: Vector database can be distributed

## Future Architecture Considerations

### 1. Database Evolution

**Vector Database Upgrade**: Consider specialized vector databases (Pinecone, Weaviate)
**Caching Layer**: Redis for frequent similarity searches
**Database Persistence**: PostgreSQL for metadata and user sessions

### 2. AI/ML Enhancements

**Model Fine-tuning**: Custom embedding models for clinical terminology
**Multi-language Support**: Embeddings for non-English content
**Advanced NLP**: Named entity recognition and clinical concept extraction

### 3. Platform Extensions

**API-First Design**: RESTful API for third-party integrations
**Microservices**: Decompose into specialized services
**Real-time Processing**: WebSocket support for live updates

This architecture provides a robust, scalable foundation for automated MDD generation with clear separation of concerns, comprehensive error handling, and efficient AI-powered semantic matching.