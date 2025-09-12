"""
Numpy-based Matching Engine - FAISS-free implementation
Provides the same functionality as EnhancedMatchingEngine but uses pure numpy for compatibility
"""
import os
import json
import pickle
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from utils.match_thresholds import THRESHOLDS
from utils.field_aliases import get_field_value as _gf

from .azure_openai_client import AzureOpenAIClient
from .mdd_file_processor import MDDFileProcessor
from utils.storage import storage
from utils.config import USE_S3
import pathlib

logger = logging.getLogger(__name__)

class NumpyMatchingEngine:
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.embeddings = None
        self.metadata = []
        self.file_processor = MDDFileProcessor()
        
    def initialize_vector_db(self) -> bool:
        """Initialize vector database with precomputed embeddings"""
        try:
            return self.load_precomputed_embeddings()
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            return False
    
    def load_precomputed_embeddings(self) -> bool:
        """Load precomputed numpy embeddings and metadata"""
        try:
            embeddings_path = 'data/precomputed_embeddings.npy'
            metadata_path = 'data/precomputed_metadata.json'

            if USE_S3:
                # Ensure both objects exist in storage
                if not (storage.exists(embeddings_path) and storage.exists(metadata_path)):
                    logger.error("Precomputed embeddings or metadata not found in storage")
                    return False
                # Download .npy to /tmp and load
                tmp_dir = pathlib.Path('/tmp/data')
                tmp_dir.mkdir(parents=True, exist_ok=True)
                emb_tmp = tmp_dir / 'precomputed_embeddings.npy'
                emb_tmp.write_bytes(storage.read_bytes(embeddings_path))
                self.embeddings = np.load(str(emb_tmp))
                # Load metadata JSON from bytes directly
                meta_bytes = storage.read_bytes(metadata_path)
                self.metadata = json.loads(meta_bytes.decode('utf-8'))
            else:
                if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
                    logger.error("Precomputed embeddings or metadata not found")
                    return False
                # Load embeddings and metadata
                self.embeddings = np.load(embeddings_path)
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"Loaded vector database: {len(self.embeddings)} OpenAI embeddings from Library of DQ specifications")
            return True
            
        except Exception as e:
            logger.error(f"Error loading precomputed embeddings: {str(e)}")
            return False
    
    def find_matches(self, target_row: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find matches using numpy-based cosine similarity"""
        logger.info(f"Rs-line56 numpy_matching_engine.py - logging target_row: {target_row}")
        try:
            if self.embeddings is None or not self.metadata:
                return []
            
            # Create embedding text
            embedding_text = self._create_enhanced_embedding_text(target_row)
            if not embedding_text or len(embedding_text) < 10:
                return []
            
            # Generate embedding for target
            target_embedding = self.client.get_embedding(embedding_text)
            if not target_embedding:
                return []
            
            # Convert to numpy array and normalize
            target_vector = np.array(target_embedding, dtype=np.float32)
            target_vector = target_vector / np.linalg.norm(target_vector)
            
            # Normalize reference embeddings
            normalized_embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            
            # Calculate cosine similarities
            similarities = np.dot(normalized_embeddings, target_vector)
            
            # Get top candidates
            top_indices = np.argsort(similarities)[::-1][:top_k * 2]
            
            enhanced_results = []
            reference_usage_count = {}
            
            for idx in top_indices:
                if idx >= len(self.metadata):
                    continue
                
                ref_data = self.metadata[idx]
                sim_score = float(similarities[idx])
                
                # Track reference usage for penalty calculation
                ref_id = ref_data.get('Reference Check Name', f"ref_{idx}")
                reference_usage_count[ref_id] = reference_usage_count.get(ref_id, 0) + 1
                
                # Calculate keyword overlap score
                logging.info(f'Rs-line100 numpy_matching_engine.py before keyword_score')
                keyword_score = self._calculate_keyword_overlap(target_row, ref_data)
                logging.info(f'Rs-line101 numpy_matching_engine.py after keyword_score {keyword_score}')
                
                # Enhanced hybrid scoring: 50% cosine + 50% keyword overlap
                base_score = 0.50 * sim_score + 0.50 * keyword_score
                
                # Apply penalty for reused references
                reuse_penalty = min(0.15, reference_usage_count[ref_id] * 0.05)
                final_score = base_score * (1 - reuse_penalty)
                
                # Basic threshold classification
                classification = self._classify_match_enhanced(final_score)
                
                # Skip very low confidence matches
                if final_score < 0.05:
                    continue
                
                # Generate detailed match explanation
                match_explanation = self._generate_enhanced_match_explanation(
                    target_row, ref_data, final_score, sim_score, keyword_score
                )
                
                result = {
                    'match_found': True,
                    'is_match_found': 'YES',
                    'confidence_score': final_score,
                    'cosine_similarity': sim_score,
                    'keyword_overlap': keyword_score,
                    'match_classification': classification,
                    **ref_data,
                    'match_reason': f"Hybrid score: {final_score:.3f} (cosine: {sim_score:.3f}, keywords: {keyword_score:.3f})",
                    'match_explanation': match_explanation
                }
                enhanced_results.append(result)
            
            # Sort by final score
            enhanced_results.sort(key=lambda x: x['confidence_score'], reverse=True)
            
            return enhanced_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            return []
    
    def _create_enhanced_embedding_text(self, row: Dict[str, Any]) -> str:
        """Create embedding text with enhanced structure and key components"""
        components = []
        
        # Extract structured components
        dq_name = _gf(row, 'dq_name')
        query_text = _gf(row, 'query_text')
        form_name = _gf(row, 'form_name')
        visit_name = _gf(row, 'visit_name')
        
        # Create structured string with domain tags
        if dq_name:
            components.append(f"CHECK:{dq_name}")
        if query_text:
            cleaned_query = self._remove_generic_phrases(query_text)
            if cleaned_query:
                components.append(f"QUERY:{cleaned_query}")
        if form_name:
            # Add domain context for laboratory data
            if any(term in form_name.lower() for term in ['lab', 'laboratory', 'lb']):
                components.append(f"FORM:{form_name} domain:LB")
            elif any(term in form_name.lower() for term in ['ae', 'adverse', 'event']):
                components.append(f"FORM:{form_name} domain:AE")
            elif any(term in form_name.lower() for term in ['vs', 'vital', 'sign']):
                components.append(f"FORM:{form_name} domain:VS")
            else:
                components.append(f"FORM:{form_name}")
        if visit_name:
            components.append(f"VISIT:{visit_name}")
        
        # Add description content with enhanced cleaning
        description = _gf(row, 'dq_description')
        if description:
            cleaned_desc = self._remove_generic_phrases(description)
            if cleaned_desc:
                if 'missing' in cleaned_desc.lower():
                    components.append(f"DESC:{cleaned_desc} check_type:missing_value")
                elif 'duplicate' in cleaned_desc.lower():
                    components.append(f"DESC:{cleaned_desc} check_type:duplicate")
                else:
                    components.append(f"DESC:{cleaned_desc}")
        
        return " ".join(components) if components else ""
    
    def _gf(self, row: Dict[str, Any], field: str) -> str:
        """Extract first non-empty value from field options"""
        value = row.get(field, '')
        if value and str(value).strip() and str(value).strip().lower() not in ['nan', 'none', 'null']:
            return str(value).strip()
        return ""
    
    def _extract_field_value(self, row: Dict[str, Any], field_options: list) -> str:
        """Extract first non-empty value from field options"""
        for field in field_options:
            value = row.get(field, '')
            if value and str(value).strip() and str(value).strip().lower() not in ['nan', 'none', 'null']:
                return str(value).strip()
        return ""
    
    def _remove_generic_phrases(self, text: str) -> str:
        """Remove generic phrases and verbose prefixes that reduce embedding quality"""
        if not text:
            return ""
        
        # Remove common generic phrases
        generic_phrases = [
            "query should fire",
            "dm review:",
            "please update",
            "please contact",
            "please review",
            "note:",
            "warning:",
            "error:",
        ]
        
        cleaned = text.lower()
        for phrase in generic_phrases:
            cleaned = cleaned.replace(phrase, "")
        
        return cleaned.strip()
    
    def _calculate_keyword_overlap(self, target_row: Dict[str, Any], ref_data: Dict[str, Any]) -> float:
        """Calculate keyword overlap score using Jaccard similarity"""
        logging.info(f'Rs-line221 numpy_matching_engine.py before target_row {target_row}')
        logging.info(f'Rs-line222 numpy_matching_engine.py before ref_data {ref_data}')
        target_text = self._create_reference_text(target_row)
        ref_text = self._create_reference_text(ref_data)
        logging.info(f'Rs-line225 numpy_matching_engine.py before target_text {target_text}')
        logging.info(f'Rs-line226 numpy_matching_engine.py before ref_text {ref_text}')
        if not target_text or not ref_text:
            return 0.0
        
        # Tokenize and normalize
        target_words = set(word.lower().strip('.,!?') for word in target_text.split() if len(word) > 2)
        ref_words = set(word.lower().strip('.,!?') for word in ref_text.split() if len(word) > 2)
        
        # Remove stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'way', 'she', 'use', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye'}
        target_words = target_words - stop_words
        ref_words = ref_words - stop_words
        
        if not target_words or not ref_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(target_words & ref_words)
        union = len(target_words | ref_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _create_reference_text(self, ref_data: Dict[str, Any]) -> str:
        """Create reference text for keyword comparison"""
        components = []
        
        # Add key fields for comparison
        desc = _gf(ref_data, 'dq_description')
        query = _gf(ref_data, 'query_text')
        form = _gf(ref_data, 'form_name')
        
        if desc:
            components.append(desc)
        if query:
            components.append(query)
        if form:
            components.append(form)
        logging.info(f'Rs-line261 numpy_matching_engine.py components {components}')
        return " ".join(components)
    
    
    def _classify_match_enhanced(self, score: float) -> str:
        """Match classification per updated thresholds: Excellent ≥60%, Good ≥40%, Moderate ≥20%, else Weak"""
        if score >= THRESHOLDS["excellent"]:
            return "Excellent Match"
        elif score >= THRESHOLDS["good"]:
            return "Good Match"
        elif score >= THRESHOLDS["moderate"]:
            return "Moderate Match"
        else:
            return "Weak Match"
    
    def _generate_enhanced_match_explanation(self, target_row: Dict[str, Any], ref_data: Dict[str, Any], 
                                           final_score: float, cosine_sim: float, keyword_overlap: float) -> str:
        """Generate detailed match explanation with comprehensive business context"""
        try:
            # Extract key fields for detailed analysis
            target_form = target_row.get('Primary Form Name', 'N/A')
            ref_form = ref_data.get('Primary Form Name', 'N/A')
            ref_sponsor = ref_data.get('Reference Sponsor', ref_data.get('Origin Study', 'Unknown'))
            
            # Build comprehensive explanation
            explanation_parts = []
            
            # Classification and confidence breakdown
            classification = self._classify_match_enhanced(final_score)
            explanation_parts.append(f"**{classification}** ({final_score:.1%} confidence)")
            
            # Scoring methodology explanation
            explanation_parts.append(f"Hybrid Score: {cosine_sim:.1%} semantic + {keyword_overlap:.1%} keyword overlap")
            
            # Semantic similarity analysis
            if cosine_sim >= 0.80:
                explanation_parts.append("✓ Very high semantic similarity - check logic nearly identical")
            elif cosine_sim >= 0.60:
                explanation_parts.append("✓ Strong semantic similarity - validation concepts align well")
            elif cosine_sim >= 0.40:
                explanation_parts.append("✓ Moderate semantic similarity - related validation patterns")
            else:
                explanation_parts.append("⚠ Lower semantic similarity - different logic but some alignment")
            
            # Keyword overlap analysis
            if keyword_overlap >= 0.60:
                explanation_parts.append("✓ High keyword overlap - many shared clinical terms")
            elif keyword_overlap >= 0.40:
                explanation_parts.append("✓ Good keyword overlap - several matching concepts")
            elif keyword_overlap >= 0.20:
                explanation_parts.append("✓ Some keyword overlap - related terminology")
            else:
                explanation_parts.append("⚠ Limited keyword overlap - different terminology")
            
            # Domain analysis
            if target_form != 'N/A' and ref_form != 'N/A':
                if target_form.lower() == ref_form.lower():
                    explanation_parts.append(f"✓ Exact form match: Both apply to '{target_form}'")
                else:
                    explanation_parts.append(f"⚠ Different forms: '{target_form}' vs '{ref_form}'")
            
            # Reference source context
            explanation_parts.append(f"Reference: {ref_sponsor} study validation library")
            
            # Business recommendation
            if final_score >= 0.60:
                explanation_parts.append("💡 **Recommendation**: Excellent reuse candidate")
            elif final_score >= 0.40:
                explanation_parts.append("💡 **Recommendation**: Good reuse candidate")
            elif final_score >= 0.25:
                explanation_parts.append("💡 **Recommendation**: Consider for adaptation")
            elif final_score >= 0.15:
                explanation_parts.append("💡 **Recommendation**: Review for insights")
            else:
                explanation_parts.append("💡 **Recommendation**: Low confidence match")
            
            return " | ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating enhanced match explanation: {str(e)}")
            return f"Match found with {final_score:.1%} confidence (semantic: {cosine_sim:.1%}, keywords: {keyword_overlap:.1%})"
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if self.embeddings is None or not self.metadata:
            return {
                'total_vectors': 0,
                'vector_dimension': 0,
                'metadata_records': 0,
                'status': 'Not initialized'
            }
        
        return {
            'total_vectors': len(self.embeddings),
            'vector_dimension': self.embeddings.shape[1] if len(self.embeddings.shape) > 1 else 0,
            'metadata_records': len(self.metadata),
            'status': 'Ready'
        }
    
    def is_initialized(self) -> bool:
        """Check if the matching engine is initialized"""
        return self.embeddings is not None and len(self.metadata) > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matching statistics (alias for get_database_stats)"""
        return self.get_database_stats()
    
    def initialize_vector_db_with_files(self, database_files: List[str] = None) -> bool:
        """Initialize vector database with database files (compatibility method)"""
        if database_files:
            return self.rebuild_vector_database(database_files)
        return self.initialize_vector_db()

    def rebuild_vector_database(self, mdd_files: List[str]) -> bool:
        """Rebuild vector database from specified MDD files"""
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"Starting vector database rebuild with {len(mdd_files)} files")
            
            # Clear existing data
            self.embeddings = None
            self.metadata = []
            
            # Use the file processor to parse all MDD files
            from .mdd_file_processor import MDDFileProcessor
            file_processor = MDDFileProcessor()
            
            all_records = []
            for file_path in mdd_files:
                logger.info(f"Processing file: {file_path}")
                try:
                    records = file_processor.parse_reference_mdd(file_path)
                    # Add source file information
                    filename = os.path.basename(file_path)
                    for record in records:
                        record['source_file'] = filename
                    all_records.extend(records)
                    logger.info(f"Added {len(records)} records from {filename}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
            
            if not all_records:
                logger.error("No records found in any MDD files")
                return False
            
            logger.info(f"Total records collected: {len(all_records)}")
            logger.info(f'Rs-line406 - logging all_records: {len(all_records),all_records[:10]}')
            # Generate embeddings for all records
            embedding_texts = []
            for record in all_records:
                embedding_text = self._create_enhanced_embedding_text(record)
                embedding_texts.append(embedding_text)
            
            logger.info("Generating embeddings...")
            embeddings = self.client.get_embeddings_batch(embedding_texts)
            logger.info(f'Rs-line415 - logging embeddings: {len(embeddings),embeddings[:10]}')
            # Filter out failed embeddings
            valid_embeddings = []
            valid_metadata = []
            for i, embedding in enumerate(embeddings):
                if embedding is not None:
                    valid_embeddings.append(embedding)
                    valid_metadata.append(all_records[i])
            logger.info(f"Total valid embeddings: {len(valid_embeddings)}")
            if not valid_embeddings:
                logger.error("No valid embeddings generated")
                return False
            logger.info(f"Total valid metadata: {len(valid_metadata)}")
            # Convert to numpy array and store
            self.embeddings = np.array(valid_embeddings)
            self.metadata = valid_metadata

            logger.info(f"Successfully rebuilt vector database with {len(valid_embeddings)} embeddings")
            
            # Save the rebuilt database
            self._save_precomputed_embeddings()
            
            return True
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error rebuilding vector database: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def _save_precomputed_embeddings(self):
        """Save precomputed embeddings to disk"""
        try:
            import pickle
            import json
            import logging
            logger = logging.getLogger(__name__)
            
            if self.embeddings is None or len(self.metadata) == 0:
                logger.error("No embeddings or metadata to save")
                return False
            
            # Create data directory if it doesn't exist
            data_dir = 'data'
            os.makedirs(data_dir, exist_ok=True)
            
            # Save embeddings
            # embeddings_path = os.path.join(data_dir, 'embeddings.npy')
            # np.save(embeddings_path, self.embeddings)
            
            # # Save metadata
            # metadata_path = os.path.join(data_dir, 'metadata.pkl')
            # with open(metadata_path, 'wb') as f:
            #     pickle.dump(self.metadata, f)
            
            # Save embeddings
            embeddings_path = os.path.join(data_dir, 'precomputed_embeddings.npy')
            if USE_S3:
                tmp_dir = pathlib.Path('/tmp/data')
                tmp_dir.mkdir(parents=True, exist_ok=True)
                emb_tmp = tmp_dir / 'precomputed_embeddings.npy'
                np.save(str(emb_tmp), self.embeddings)
                with open(emb_tmp, 'rb') as _f:
                    storage.write_bytes('data/precomputed_embeddings.npy', _f.read())
            else:
                np.save(embeddings_path, self.embeddings)
            
            # Save metadata
            metadata_path = os.path.join(data_dir, 'precomputed_metadata.json')
            if USE_S3:
                storage.write_bytes('data/precomputed_metadata.json', json.dumps(self.metadata).encode('utf-8'))
            else:
                with open(metadata_path, 'w') as f:
                    json.dump(self.metadata, f)
            
            logger.info(f"Saved {len(self.embeddings)} embeddings and metadata to {data_dir}")
            return True
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error saving precomputed embeddings: {str(e)}")
            return False
    
    def append_vector_database(self, mdd_files: List[str]) -> bool:
        """Append new vectors from the supplied MDD files to the existing vector database.
        It loads the existing index/embeddings if necessary and only processes files
        that were not part of the database previously (identified via `source_file`)."""
        try:
            logger = logging.getLogger(__name__)

            # Ensure the current database is loaded (if it exists)
            logger.info(f"Before loading: is_initialized={self.is_initialized()}, metadata count={len(self.metadata) if self.metadata else 0}")
            
            # Always load the latest metadata to ensure we have the most recent data
            try:
                # Check if precomputed files exist before trying to load them
                if USE_S3:
                    have_data = storage.exists('data/precomputed_embeddings.npy') and storage.exists('data/precomputed_metadata.json')
                else:
                    emb_path = os.path.join('data', 'precomputed_embeddings.npy')
                    meta_path = os.path.join('data', 'precomputed_metadata.json')
                    have_data = os.path.exists(emb_path) and os.path.exists(meta_path)
                if have_data:
                    self.load_precomputed_embeddings()
                    logger.info(f"After loading: metadata count={len(self.metadata) if self.metadata else 0}")
                else:
                    logger.info("No existing metadata files found - this appears to be a first-time run")
                    # Initialize empty structures for first run
                    self.embeddings = None
                    self.metadata = []
                    self.index = None
            except Exception as e:
                logger.error(f"Error loading precomputed embeddings: {str(e)}")
                # If loading fails, initialize empty structures
                if self.metadata is None:
                    self.metadata = []
                if self.embeddings is None:
                    self.embeddings = None  # Will be initialized later
                self.index = None  # Will be initialized later

            # Extract just the filenames (not paths) from metadata records
            processed_files = {rec.get('source_file') for rec in self.metadata}
            logger.info(f"Processed files in metadata: {processed_files}")
            
            # Compare basenames on both sides for consistent matching
            new_files = []
            for f in mdd_files:
                basename = os.path.basename(f)
                logger.info(f"Checking file: {basename}")
                if basename not in processed_files:
                    new_files.append(f)
                else:
                    logger.info(f"Skipping already processed file: {basename}")

            if not new_files:
                logger.info("No new MDD files detected – vector database is already up-to-date")
                return True

            logger.info(f"Appending vectors from {len(new_files)} new MDD files")

            from .mdd_file_processor import MDDFileProcessor
            file_processor = MDDFileProcessor()

            new_records: List[Dict[str, Any]] = []
            for file_path in new_files:
                try:
                    records = file_processor.parse_reference_mdd(file_path)
                    # logger.info(f"Rs-line811 enhanced_matching_engine.py append_vector_database - logging records: {records}")
                    filename = os.path.basename(file_path)
                    # logger.info(f"Rs-line813 enhanced_matching_engine.py append_vector_database - logging filename: {filename}")
                    for rec in records:
                        rec['source_file'] = filename
                        # logger.info(f"Rs-line816 enhanced_matching_engine.py append_vector_database - logging rec: {rec}")
                    new_records.extend(records)
                    logger.info(f"Added {len(records)} records from {filename}")
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")

            if not new_records:
                logger.warning("No valid new records found to append")
                return True  # Nothing new to add, but not a failure

            # Deduplicate against existing records and within the new batch
            # Signature = JSON dump of the row (lower-cased) excluding volatile keys
            import json
            
            
            def _normalise_value(val: Any) -> str:
                """Return a canonical string representation for comparison"""
                if val is None:
                    return ''
                # Ensure consistent string representation
                return str(val).strip().lower()
            
            def make_signature(rec: Dict[str, Any]) -> str:
                """Create a unique signature for a record based on its content
                   This is used to detect duplicates across different files"""
                # Only use keys that identify the actual content
                # Keys that uniquely identify the check irrespective of row-specific noise
                content_keys = [
                    'sponsor_name', 'study_id', 'dq_name', 'dq_description',
                    'query_text', 'query_target', 'primary_dataset'
                ]
                
                # Build signature from important fields only
                signature_parts = []
                for key in content_keys:
                    if key in rec:
                        signature_parts.append(f"{key}:{_normalise_value(rec[key])}")
                
                # Create a consistent signature string
                signature = "|".join(sorted(signature_parts))
                # logger.info(f"Created signature: {signature[:100]}...")
                return signature
            # logger.info(f"Rs-line847 enhanced_matching_engine.py append_vector_database - before make_signature for metadata")
            existing_keys = {make_signature(r) for r in self.metadata}
            # logger.info(f"Rs-line849 enhanced_matching_engine.py append_vector_database - after make_signature for metadata")
            # logger.info(f"Rs-line849 enhanced_matching_engine.py append_vector_database - logging existing_keys: {existing_keys}")
            unique_records: List[Dict[str, Any]] = []
            batch_keys = set()
            for rec in new_records:
                sig = make_signature(rec)
                logger.info(f"Rs-line854 enhanced_matching_engine.py append_vector_database - logging sig: {sig}")
                # logger.info(f"Rs-line855 enhanced_matching_engine.py append_vector_database - logging existing_keys: {existing_keys}")
                # logger.info(f"Rs-line856 enhanced_matching_engine.py append_vector_database - logging batch_keys: {batch_keys}")
                if sig in existing_keys or sig in batch_keys:
                    logger.info(f"Rs-line858 enhanced_matching_engine.py append_vector_database - duplicate detected {sig}")
                    continue  # duplicate detected
                unique_records.append(rec)
                batch_keys.add(sig)

            if not unique_records:
                logger.info("All records in the new files are duplicates; nothing to append")
                return True

            # Generate embeddings only for unique records
            # logger.info(f"Rs-line866 enhanced_matching_engine.py append_vector_database - logging unique_records: {unique_records}")
            embedding_texts = [self._create_enhanced_embedding_text(rec) for rec in unique_records]
            embeddings = self.client.get_embeddings_batch(embedding_texts)
            logger.info(f"Rs-line866 enhanced_matching_engine.py append_vector_database - logging embeddings: {len(embeddings)}")
            valid_embeddings = []
            valid_metadata = []
            for i, emb in enumerate(embeddings):
                if emb is not None:
                    valid_embeddings.append(emb)
                    # Use the corresponding unique record to maintain alignment
                    valid_metadata.append(unique_records[i])

            if not valid_embeddings:
                logger.error("Failed to generate embeddings for new records")
                return False

            new_emb_arr = np.array(valid_embeddings, dtype=np.float32)
            import faiss
            faiss.normalize_L2(new_emb_arr)

            # Create or reuse FAISS index
            if self.index is None:
                self.index = faiss.IndexFlatIP(new_emb_arr.shape[1])
            elif self.index.d != new_emb_arr.shape[1]:
                logger.error("Embedding dimension mismatch between existing index and new embeddings")
                return False

            # Append to FAISS index and in-memory structures
            self.index.add(new_emb_arr)
            if self.embeddings is None:
                self.embeddings = new_emb_arr
            else:
                self.embeddings = np.vstack([self.embeddings, new_emb_arr])
            self.metadata.extend(valid_metadata)
            
            # Persist updated database
            index_path = 'data/faiss_index.bin'
            if USE_S3:
                tmp_dir = pathlib.Path('/tmp/data'); tmp_dir.mkdir(parents=True, exist_ok=True)
                index_path = str(tmp_dir / 'faiss_index.bin')
            faiss.write_index(self.index, index_path)
            if USE_S3:
                with open(index_path, 'rb') as _f:
                    storage.write_bytes('data/faiss_index.bin', _f.read())
            self._save_precomputed_embeddings()
            logger.info(f"Appended {len(valid_embeddings)} new embeddings. Total vectors: {self.index.ntotal}")
            return True
        except Exception as e:
            logger.error(f"Error appending to vector database: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
