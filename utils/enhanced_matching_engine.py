"""
Enhanced Matching Engine with improved semantic search algorithm
Implements 4 key enhancements:
1. Improved text cleaning before embedding
2. Hybrid scoring model (cosine + keyword overlap)
3. Lowered threshold for better match detection
4. Domain pre-filtering for accuracy
"""

import numpy as np
import faiss
import pickle
import os
import json
import csv
import difflib
from datetime import datetime
from utils.match_thresholds import THRESHOLDS
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from .azure_openai_client import AzureOpenAIClient
from utils.field_aliases import get_field_value as _gf
import re
from utils.storage import storage
from utils.config import USE_S3,S3_BUCKET
import pathlib

logger = logging.getLogger(__name__)
print(f"Rs-line26 enhanced_matching_engine.py - logging logger: {logger}")

class EnhancedMatchingEngine:
    def __init__(self, client: AzureOpenAIClient):
        self.client = client
        self.index = None
        self.metadata = []
        self.embeddings = None
        # Map from normalized sponsor -> list of record indices
        self.sponsor_to_indices: Dict[str, List[int]] = {}
        # Safety cap on number of candidates to fully score for same-sponsor subset
        self.same_sponsor_topn_cap: int = 50
        # Cache for variable configuration CSV
        self._var_config_cache: Dict[str, Any] = {}
        
    def initialize_vector_db(self) -> bool:
        """Initialize vector database with precomputed embeddings"""
        try:
            # Try to load existing database first
            if self.load_precomputed_embeddings():
                logger.info("Loaded existing vector database")
                return True
            
            logger.info("No existing database found, building new one...")
            return self.build_vector_database()
            
        except Exception as e:
            logger.error(f"Failed to initialize vector database: {str(e)}")
            return False
    
    def load_precomputed_embeddings(self) -> bool:
        """Load precomputed FAISS index and metadata"""
        try:
            logger.info(f"Rs-line57 enhanced_matching_engine.py - logging USE_S3: {USE_S3}")
            if USE_S3:
                # Download to /tmp and load
                logger.info(f"Rs-line60 enhanced_matching_engine.py - logging USE_S3 INSIDE: {USE_S3}")
                # S3 connectivity and top-level "folders" diagnostic
                try:
                    keys = storage.list_keys("")  # list all keys at bucket root
                    top_folders = sorted({k.split('/')[0] for (k, _) in keys if '/' in k})
                    logger.info(f"S3 connectivity OK (bucket={S3_BUCKET}). Top-level folders: {top_folders}")
                except Exception as e:
                    logger.error(f"S3 connectivity check failed: {e}")
                tmp_dir = pathlib.Path('/tmp/data')
                tmp_dir.mkdir(parents=True, exist_ok=True)
                if not storage.exists('data/faiss_index.bin'):
                    return False
                index_tmp = tmp_dir / 'faiss_index.bin'
                index_tmp.write_bytes(storage.read_bytes('data/faiss_index.bin'))
                self.index = faiss.read_index(str(index_tmp))

                emb_key = 'data/embeddings.npy'
                if storage.exists(emb_key):
                    emb_tmp = tmp_dir / 'embeddings.npy'
                    emb_tmp.write_bytes(storage.read_bytes(emb_key))
                    self.embeddings = np.load(str(emb_tmp))

                meta_key = 'data/metadata.pkl'
                if storage.exists(meta_key):
                    import pickle as _pickle
                    self.metadata = _pickle.loads(storage.read_bytes(meta_key))
            else:
                if not os.path.exists('data/faiss_index.bin'):
                    return False
                # Load FAISS index
                self.index = faiss.read_index('data/faiss_index.bin')
                # Load embeddings
                if os.path.exists('data/embeddings.npy'):
                    self.embeddings = np.load('data/embeddings.npy')
                # Load metadata
                if os.path.exists('data/metadata.pkl'):
                    with open('data/metadata.pkl', 'rb') as f:
                        self.metadata = pickle.load(f)

            # Ensure embeddings are L2-normalized for cosine/IP equivalence
            try:
                if self.embeddings is not None:
                    faiss.normalize_L2(self.embeddings)
            except Exception:
                pass

            # Build sponsor index map for same-sponsor queries
            try:
                self._build_sponsor_index_map()
            except Exception as _e:
                logger.warning(f"Failed to build sponsor index map: {str(_e)}")

            logger.info(f"Loaded vector database: {self.index.ntotal} vectors, {len(self.metadata)} metadata records")
            return True
        except Exception as e:
            logger.error(f"Error loading precomputed embeddings: {str(e)}")
            return False
    
    def build_vector_database(self) -> bool:
        """Build vector database from MDD files"""
        logger.info("Building vector database from MDD files...")
        
        # This would be called during initialization if no precomputed database exists
        # For now, return False to indicate no database available
        return False
    
    def find_matches(self, target_row: Dict[str, Any], top_k: int = 5, sponsor_filter: Optional[str] = None, scope: str = 'across') -> List[Dict[str, Any]]:
        """Find matches using enhanced algorithm with all 4 improvements"""
        if not self.index or not self.metadata:
            logger.warning("Vector database not initialized")
            return []
        
        try:
            # ENHANCEMENT 1: Clean input text before embedding
            # logger.info(f"Rs-line84 enhanced_matching_engine.py - logging target_row: {target_row}")
            embedding_text = self._create_enhanced_embedding_text(target_row)
            if not embedding_text or len(embedding_text) < 10:
                return []
        
            # Generate embedding for target
            # logger.info(f"Rs-line92 enhanced_matching_engine.py - logging embedding_text: {embedding_text}")
            target_embedding = self.client.get_embedding(embedding_text)
            if not target_embedding:
                return []
            # logger.info(f"Rs-line95 enhanced_matching_engine.py - logging target_embedding: {len(target_embedding)}")
            # Search FAISS index
            target_vector = np.array(target_embedding, dtype=np.float32).reshape(1, -1)
            # logger.info(f"Rs-line97 enhanced_matching_engine.py - before normalize target_vector: {len(target_vector)}")
            faiss.normalize_L2(target_vector)
            # logger.info(f"Rs-line99 enhanced_matching_engine.py - after normalize target_vector: {len(target_vector)}")
            
            # Two paths: same-sponsor subset search (exact) vs across-sponsors FAISS search (exact)
            if scope == 'same' and sponsor_filter:
                # Clean the user-provided sponsor, then group by substring match both ways
                filter_key = self._normalize_sponsor(sponsor_filter)
                logger.info(f"Rs-line145 enhanced_matching_engine.py - filter_key: {filter_key}")
                keys_view = list(self.sponsor_to_indices.keys())
                logger.info(f"Rs-line146 enhanced_matching_engine.py - sponsor_to_indices KEYS: {keys_view}")
                candidate_keys = [k for k in keys_view if filter_key and (filter_key in k or k in filter_key)]
                idxs_list: List[int] = []
                for k in candidate_keys:
                    idxs_list.extend(self.sponsor_to_indices.get(k, []))
                    logger.info(f'Rs-line152 enhanced_matching_engine.py - len idxs_list: {k} , {len(self.sponsor_to_indices.get(k, []))}, {len(idxs_list)}')
                # Deduplicate indices in case multiple keys overlap
                idxs = sorted(set(idxs_list))
                # logger.info(f'Rs-line155 enhanced_matching_engine.py - grouped idxs: {idxs}')
                if not idxs:
                    return []
                # Embeddings are L2-normalized at build/load; compute exact cosine via matmul
                subset = self.embeddings[idxs]
                sims = subset @ target_vector.T  # shape (len(idxs), 1) ## @ -> Does multiplication of both the 2d vectors
                sims = sims.reshape(-1)
                order = np.argsort(-sims)
                # Bound Python-side work but avoid missing hits due to tiny k (cap at same_sponsor_topn_cap)
                top_n = min(len(order), min(self.same_sponsor_topn_cap, max(2 * top_k, 20)))
                order = order[:top_n]
                similarities = np.array([sims[order]], dtype=np.float32)
                indices = np.array([[idxs[i] for i in order]], dtype=np.int64)
            else:
                # Across sponsors - FAISS IndexFlatIP (exact cosine on normalized vectors)
                # similarities, indices = self.index.search(target_vector, top_k * 2)  # Get more candidates ## L2 Search
                # Search FAISS index - updated for cosine similarity
                distances, indices = self.index.search(target_vector, top_k * 2)  # Get more candidates
                
                # Convert distances to similarities (since we're using IndexFlatIP with normalized vectors)
                # similarities = 1.0 - distances  # For normalized vectors, inner product = cosine similarity
                similarities = distances  # For this index faiss.IndexFlatIP.search already returns the inner-product (= cosine-similarity).
                # logger.info(f"Rs-line101 enhanced_matching_engine.py - logging similarities: {similarities}")
                # logger.info(f"Rs-line102 enhanced_matching_engine.py - logging indices: {indices}")
            
            # SIMPLIFIED: No domain filtering - process all results directly
            enhanced_results = []
            reference_usage_count = {}  # Track how many times each reference is used
            
            for sim_score, ref_idx in zip(similarities[0], indices[0]):
                # logger.info(f"Rs-line110 enhanced_matching_engine.py - logging sim_score and ref_idx: {sim_score},{ref_idx}")
                if ref_idx >= len(self.metadata):
                    logger.warning(f"Rs-line112 enhanced_matching_engine.py - ref_idx out of bounds {len(self.metadata)},{len(ref_idx)},{ref_idx}")
                    continue
                
                ref_data = self.metadata[ref_idx]
                # logger.info(f"Rs-line115 enhanced_matching_engine.py - logging ref_data: {ref_data}")
                
                # Track reference usage for penalty calculation
                ref_id = ref_data.get('Reference Check Name', f"ref_{ref_idx}")
                # logger.info(f"Rs-line119 enhanced_matching_engine.py - logging ref_id: {ref_id}")
                reference_usage_count[ref_id] = reference_usage_count.get(ref_id, 0) + 1
                
                # Calculate keyword overlap score
                keyword_score = self._calculate_keyword_overlap(target_row, ref_data)
                # logger.info(f"Rs-line124 enhanced_matching_engine.py - logging keyword_score: {keyword_score}")
                
                # Enhanced hybrid scoring: 50% cosine + 50% keyword overlap (increased keyword weight)
                base_score = 0.50 * sim_score + 0.50 * keyword_score
                # logger.info(f"Rs-line128 enhanced_matching_engine.py - logging base_score: {base_score}")
                
                # Apply penalty for reused references (reduce score by 5% for each reuse)
                reuse_penalty = min(0.15, reference_usage_count[ref_id] * 0.02)
                final_score = base_score * (1 - reuse_penalty)
                # logger.info(f"Rs-line132 enhanced_matching_engine.py - logging final_score: {final_score}")
                
                # SIMPLIFIED: Basic threshold classification only
                classification = self._classify_match_enhanced(final_score)
                # logger.info(f"Rs-line136 enhanced_matching_engine.py - logging classification: {classification}")
                
                # Log weak/borderline matches for inspection (0.02-0.4 range)
                if 0.02 <= final_score <= 0.4:
                    logger.info(f"DEBUG: Weak match found - Score: {final_score:.3f}, Classification: {classification}")
                    logger.info(f"  Target: {target_row.get('Target Check Description', 'N/A')[:100]}...")
                    logger.info(f"  Reference: {ref_data.get('DQ description', 'N/A')[:100]}...")
                # logger.info(f"Rs-line145 enhanced_matching_engine.py - logging classification condition check: {classification != 'No Match'}")
                # REMOVED: No business logic validation or domain filtering
                if classification != "No Match":  # Only include matches above threshold
                    # Generate detailed match explanation
                    # logger.info(f"Rs-line149 enhanced_matching_engine.py - logging classification: {classification}")

                    # match_explanation = self._generate_enhanced_match_explanation(target_row, ref_data, final_score, sim_score, keyword_score)
                    # match_explanation = self._generate_match_explanation(target_row, ref_data, final_score, sim_score, keyword_score) ##TRY
                    # logger.info(f"Rs-line155 enhanced_matching_engine.py - logging match_explanation: {match_explanation}")
                    result = {
                        'match_found': True,
                        'is_match_found': 'YES',
                        'confidence_score': final_score,
                        'cosine_similarity': sim_score,
                        'keyword_overlap': keyword_score,
                        'match_classification': classification,
                        **ref_data,
                        'match_reason': f"Hybrid score: {final_score:.3f} (cosine: {sim_score:.3f}, keywords: {keyword_score:.3f})",
                        'match_explanation': '', #match_explanation
                        'ref_data': ref_data #TRY
                    }
                    enhanced_results.append(result)
                    # logger.info(f'Rs-line166 enhanced_matching_engine.py - logging result: {result}')
            # Sort by final score
            # logging.info(f'Rs-line168 enhanced_matching_engine.py - logging enhanced_results: {len(enhanced_results)}')
            enhanced_results.sort(key=lambda x: x['confidence_score'], reverse=True)
            logging.info(f'Rs-line170 enhanced_matching_engine.py - logging enhanced_results: {enhanced_results}')
            if enhanced_results:
                    enhanced_results = enhanced_results[0]  # Best match
                    enhanced_results['match_explanation'] = self._generate_match_explanation(target_row, enhanced_results['ref_data'], enhanced_results['confidence_score'], enhanced_results['cosine_similarity'], enhanced_results['keyword_overlap']) ##TRY
                    logger.info(f"Rs-line155 enhanced_matching_engine.py - logging match_explanation: {enhanced_results['match_explanation']}")
                    # Generate Tech Logic for less than 30% final score
                    if enhanced_results['confidence_score'] < 0.99:
                        # logger.info(f'Rs-line261 enhanced_matching_engine.py - ref_data {target_row.get("DQ Name")}')
                        enhanced_results['pseudo_tech_code'] = self._generate_tech_logic(target_row)
                        # generated_code = self._generate_tech_logic(target_row)
                        # Harmonize variable labels to actual variable names using CSV mapping and optional OpenAI assistance
                        # enhanced_results['pseudo_tech_code'] = self._harmonize_variables_in_pseudo_code(
                        #     generated_code,
                        #     target_row
                        # )
            del enhanced_results['ref_data']
                        
            return enhanced_results
        
        except Exception as e:
            logger.error(f"Error finding matches: {str(e)}")
            return []

    def _normalize_sponsor(self, name: Optional[str]) -> str:
        """Clean sponsor for matching: lowercase, replace non-alphanumerics with spaces, collapse spaces."""
        if not name:
            return ''
        s = str(name).lower().strip()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        return " ".join(s.split())

    def _build_sponsor_index_map(self) -> None:
        """Build mapping from normalized sponsor -> list of indices in metadata/embeddings."""
        sponsor_map: Dict[str, List[int]] = {}
        for i, rec in enumerate(self.metadata or []):
            sp = _gf(rec, 'sponsor_name') or rec.get('sponsor_name') or _gf(rec, 'source_file') or ''
            key = self._normalize_sponsor(sp)
            if not key:
                continue
            sponsor_map.setdefault(key, []).append(i)
        self.sponsor_to_indices = sponsor_map
        logger.info(f"Built sponsor_to_indices for {len(self.sponsor_to_indices)} sponsors")
    
    def _create_enhanced_embedding_text(self, row: Dict[str, Any]) -> str:
        """Create embedding text with enhanced structure and key components"""
        # Extract structured components for better embedding quality
        components = []
        
        # 1. DQ Name + Standard Query text + Primary Form/Visit for structured embedding
        sponsor_name = _gf(row, 'sponsor_name')
        dq_name = _gf(row, 'dq_name')
        query_text = _gf(row, 'query_text')
        form_name = _gf(row, 'form_name')
        visit_name = _gf(row, 'visit_name')
        
        # 2. Create structured string with domain tags for better granularity
        # if sponsor_name:
            # components.append(f"SPONSOR:{sponsor_name}")
        # if dq_name:
            # components.append(f"CHECK:{dq_name}")
        if query_text:
            # Clean query text by removing generic phrases
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
        
        # 3. Add description content with enhanced cleaning and check type context
        description = _gf(row, 'dq_description')
        if description:
            cleaned_desc = self._remove_generic_phrases(description)
            if cleaned_desc:
                # Add check type context based on description content
                if 'missing' in cleaned_desc.lower():
                    components.append(f"DESC:{cleaned_desc} check_type:missing_value")
                elif 'range' in cleaned_desc.lower() or 'limit' in cleaned_desc.lower():
                    components.append(f"DESC:{cleaned_desc} check_type:range_check")
                elif 'consistency' in cleaned_desc.lower() or 'consistent' in cleaned_desc.lower():
                    components.append(f"DESC:{cleaned_desc} check_type:consistency")
                else:
                    components.append(f"DESC:{cleaned_desc}")
        
        # 4. Add any pseudo code or logic components
        logic = _gf(row, 'pseudo_code')
        if logic:
            cleaned_logic = self._remove_generic_phrases(logic)
            if cleaned_logic:
                components.append(f"LOGIC:{cleaned_logic}")
        # logger.info(f"Rs-line217 enhanced_matching_engine.py - logging components: {components}")
        return ' '.join(components) if components else self._fallback_text_extraction(row)
    
    def _extract_field_value(self, row: Dict[str, Any], field_options: list) -> str:
        """Extract first non-empty value from field options"""
        for field in field_options:
            value = row.get(field, '')
            if value and str(value).strip() != '' and str(value).strip().lower() not in ['n/a', 'na', '']:
                return str(value).strip()
        return ''
    
    def _remove_generic_phrases(self, text: str) -> str:
        """Remove generic phrases and verbose prefixes that reduce embedding quality"""
        if not text:
            return ''
        
        import re
        
        text = str(text).lower()
        
        # Enhanced prefix cleaning using regex
        text = self._clean_prefixes(text)
        
        # Remove common generic phrases
        generic_phrases = [
            'please verify', 'query should fire', 'dm review:', 'data management review',
            'check that', 'ensure that', 'verify that', 'confirm that',
            'should be', 'must be', 'needs to be', 'required to be',
            'check if the value of', 'value of'
        ]
        
        for phrase in generic_phrases:
            text = text.replace(phrase, '')
        
        # Clean up extra whitespace
        text = ' '.join(text.split())
        
        return text if len(text) > 10 else ''  # Return only if meaningful content remains
    
    def _clean_prefixes(self, text: str) -> str:
        """Clean verbose prefixes from text using regex"""
        import re
        
        # Strip common verbose prefixes
        prefix_patterns = [
            r"please verify whether\.\.\.?\s*",
            r"check if the value of\.\.\.?\s*", 
            r"verify\s*",
            r"check\s*",
            r"ensure\s*",
            r"confirm\s*"
        ]
        
        for pattern in prefix_patterns:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _fallback_text_extraction(self, row: Dict[str, Any]) -> str:
        """Fallback method for text extraction using flexible patterns"""
        texts = []
        field_patterns = [
            ['description', 'desc'],
            ['query', 'text'],
            ['name', 'id'],
            ['form', 'dataset'],
            ['visit', 'timepoint'],
            ['code', 'logic', 'pseudo']
        ]
        
        for pattern_group in field_patterns:
            for field_name, field_value in row.items():
                if any(pattern.lower() in str(field_name).lower() for pattern in pattern_group):
                    if field_value and str(field_value).strip() != '':
                        texts.append(str(field_value))
                        break
        logger.info(f"Rs-line293 enhanced_matching_engine.py - _fallback_text_extraction: {texts}")
        return ' '.join(texts)

    # # ========================= Variable Harmonization (Label -> Variable Name) =========================
    # def _get_default_var_config_path(self) -> str:
    #     """Resolve the default path of the Domain Variable Config CSV inside the repo."""
    #     # Attempt project-relative default
    #     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    #     candidate = os.path.join(
    #         base_dir,
    #         'Check_logic_collated_MDD',
    #         'Preconf',
    #         'Domain_variable_config_05Sep2025_08_28_17.csv'
    #     )
    #     return candidate

    # def _load_variable_config(self, csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    #     """Load and cache the variable configuration CSV.
    #     Returns list of rows (dicts) with normalized keys.
    #     """
    #     if not csv_path:
    #         csv_path = self._get_default_var_config_path()

    #     if csv_path in self._var_config_cache:
    #         return self._var_config_cache[csv_path]

    #     rows: List[Dict[str, Any]] = []
    #     try:
    #         if not os.path.exists(csv_path):
    #             logger.warning(f"Variable config CSV not found at: {csv_path}")
    #             self._var_config_cache[csv_path] = rows
    #             return rows

    #         with open(csv_path, 'r', encoding='utf-8-sig', newline='') as f:
    #             reader = csv.DictReader(f)
    #             # Normalize header mapping
    #             header_map = {h: h.lower().strip() for h in (reader.fieldnames or [])}

    #             def pick(col_patterns: List[str]) -> Optional[str]:
    #                 for original, lower in header_map.items():
    #                     if any(p in lower for p in col_patterns):
    #                         return original
    #                 return None

    #             domain_col = pick(['domain', 'dataset'])
    #             var_name_col = pick(['variable name', 'variablename', 'varname', 'variable_name'])
    #             var_label_col = pick(['variable label', 'label'])
    #             data_cat_col = pick(['data category'])

    #             if not var_name_col or not var_label_col:
    #                 logger.warning("Variable config CSV missing required columns (variable name/label)")
    #             for r in reader:
    #                 rows.append({
    #                     'domain': (r.get(domain_col) or '').strip() if domain_col else '',
    #                     'var_name': (r.get(var_name_col) or '').strip() if var_name_col else '',
    #                     'var_label': (r.get(var_label_col) or '').strip() if var_label_col else '',
    #                     'data_category': (r.get(data_cat_col) or '').strip() if data_cat_col else ''
    #                 })

    #         self._var_config_cache[csv_path] = rows
    #     except Exception as e:
    #         logger.error(f"Failed to load variable config CSV: {e}")
    #         self._var_config_cache[csv_path] = rows
    #     return rows

    # def _filter_candidates_by_context(self, all_rows: List[Dict[str, Any]], domain: str, data_category: str) -> List[Dict[str, str]]:
    #     """Filter variable candidates to the appropriate domain and optionally data_category."""
    #     domain_norm = (domain or '').strip().upper()
    #     data_cat_norm = (data_category or '').strip().upper()
    #     candidates: List[Dict[str, str]] = []
    #     for r in all_rows:
    #         d = (r.get('domain') or '').strip().upper()
    #         dc = (r.get('data_category') or '').strip().upper()
    #         if domain_norm and d and domain_norm != d:
    #             continue
    #         if data_cat_norm and dc and data_cat_norm != dc:
    #             continue
    #         name = r.get('var_name') or ''
    #         label = r.get('var_label') or ''
    #         if name and label:
    #             candidates.append({'name': name, 'label': label})
    #     # Deduplicate by name/label
    #     seen = set()
    #     unique: List[Dict[str, str]] = []
    #     for c in candidates:
    #         key = (c['name'].upper(), c['label'].upper())
    #         if key not in seen:
    #             seen.add(key)
    #             unique.append(c)
    #     return unique

    # def _label_regex(self, label: str) -> re.Pattern:
    #     """Build a case-insensitive regex for a label phrase, allowing flexible whitespace and punctuation."""
    #     # Escape, then loosen spaces/hyphens/underscores
    #     escaped = re.escape(label)
    #     # Replace escaped spaces with flexible whitespace
    #     escaped = escaped.replace(r"\ ", r"\\s+")
    #     # Allow hyphens/underscores to be optional separators
    #     escaped = escaped.replace(r"\-", r"[-_]?")
    #     pattern = r"(?<![A-Z0-9_])" + escaped + r"(?![A-Z0-9_])"
    #     return re.compile(pattern, flags=re.IGNORECASE)

    # def _apply_direct_label_replacements(self, code: str, candidates: List[Dict[str, str]]) -> Tuple[str, int]:
    #     """Replace occurrences of variable labels in code with their variable names using direct regex matching."""
    #     # Replace longer labels first to avoid partial overshadowing
    #     sorted_cands = sorted(candidates, key=lambda c: len(c['label']), reverse=True)
    #     replaced = 0
    #     out = code
    #     for c in sorted_cands:
    #         lab = c['label']
    #         var = c['name']
    #         if not lab or not var:
    #             continue
    #         rx = self._label_regex(lab)
    #         new_out, n = rx.subn(var, out)
    #         if n > 0:
    #             replaced += n
    #             out = new_out
    #     return out, replaced

    # def _apply_fuzzy_label_replacements(self, code: str, candidates: List[Dict[str, str]], max_pairs: int = 30) -> Tuple[str, int]:
    #     """Use fuzzy matching (difflib) to replace near-matching label phrases with variable names."""
    #     # Extract uppercase-ish phrases that are not clearly variables (heuristic)
    #     tokens = set(re.findall(r"[A-Za-z][A-Za-z\s]{2,}", code))  # multi-word phrases
    #     labels = [c['label'] for c in candidates]
    #     out = code
    #     replaced = 0
    #     for t in sorted(tokens, key=len, reverse=True):
    #         match = difflib.get_close_matches(t, labels, n=1, cutoff=0.86)
    #         if not match:
    #             continue
    #         lab = match[0]
    #         # Find candidate by label
    #         cand = next((c for c in candidates if c['label'] == lab), None)
    #         if not cand:
    #             continue
    #         rx = self._label_regex(lab)
    #         new_out, n = rx.subn(cand['name'], out)
    #         if n > 0:
    #             replaced += n
    #             out = new_out
    #             if replaced >= max_pairs:
    #                 break
    #     return out, replaced

    # def _openai_label_mapping(self, code: str, candidates: List[Dict[str, str]], domain: str) -> Dict[str, str]:
    #     """Ask the OpenAI client to propose label->var_name mappings present in code for logical/partial matches."""
    #     try:
    #         if not self.client or not self.client.is_available():
    #             return {}
    #         # Limit candidates to reduce prompt size
    #         limited = candidates[:80]
    #         return self.client.suggest_label_to_var_mapping(code, limited, domain)
    #     except Exception as e:
    #         logger.warning(f"OpenAI label mapping failed: {e}")
    #         return {}

    # def _harmonize_variables_in_pseudo_code(self, code: str, target_row: Dict[str, Any], csv_path: Optional[str] = None) -> str:
    #     """Replace variable labels in the generated code with actual variable names using CSV mapping and OpenAI assistance.
    #     - Filters mapping by domain and optional data category.
    #     - Applies direct, fuzzy, then OpenAI-guided replacements.
    #     """
    #     if not code:
    #         return code

    #     # Determine context
    #     domain = _gf(target_row, 'primary_dataset') or target_row.get('Domain') or _gf(target_row, 'domain') or ''
    #     data_category = _gf(target_row, 'data_category') or target_row.get('Data Category: IDRP Data Category Name') or ''

    #     # Load mapping rows and filter
    #     rows = self._load_variable_config(csv_path)
    #     if not rows:
    #         return code
    #     candidates = self._filter_candidates_by_context(rows, domain, data_category)
    #     if not candidates:
    #         candidates = self._filter_candidates_by_context(rows, domain='', data_category=data_category)
    #     if not candidates:
    #         candidates = self._filter_candidates_by_context(rows, domain='', data_category='')

    #     # 1) Direct label substitutions
    #     out, n1 = self._apply_direct_label_replacements(code, candidates)
    #     # 2) Fuzzy substitutions if needed
    #     out, n2 = self._apply_fuzzy_label_replacements(out, candidates) if n1 < 1 else (out, 0)
    #     # 3) OpenAI-guided mapping for logical matches if still minimal replacements
    #     if (n1 + n2) < 1:
    #         mapping = self._openai_label_mapping(out, candidates, domain)
    #         # Apply suggested mappings exactly
    #         for lab, var in mapping.items():
    #             try:
    #                 rx = self._label_regex(lab)
    #                 out = rx.sub(var, out)
    #             except Exception:
    #                 continue

    #     return out
    ###########################################
    def _apply_domain_prefilter(self, target_row: Dict[str, Any], similarities: np.ndarray, indices: np.ndarray) -> List[Tuple[float, int]]:
        """Apply domain pre-filtering (Enhancement 4)"""
        target_dataset = self._extract_primary_dataset(target_row)
        
        filtered_results = []
        for sim_score, ref_idx in zip(similarities, indices):
            if ref_idx >= len(self.metadata):
                continue
            
            ref_data = self.metadata[ref_idx]
            ref_dataset = self._extract_primary_dataset(ref_data)
            
            # If both have primary dataset info, only compare within same domain
            if target_dataset and ref_dataset:
                if target_dataset.lower() == ref_dataset.lower():
                    filtered_results.append((sim_score, ref_idx))
                # Skip cross-domain comparisons when domain info is available
            else:
                # Include if domain info not available
                filtered_results.append((sim_score, ref_idx))
        
        return filtered_results
    
    def _extract_primary_dataset(self, row: Dict[str, Any]) -> Optional[str]:
        """Extract primary dataset information from row"""
        dataset_patterns = ['primary dataset', 'dataset', 'domain', 'form', 'study']
        
        for column_name, value in row.items():
            if not value:
                continue
            
            column_lower = str(column_name).lower().strip()
            for pattern in dataset_patterns:
                if pattern in column_lower:
                    return str(value).strip()
        
        return None
    
    def _calculate_keyword_overlap(self, target_row: Dict[str, Any], ref_data: Dict[str, Any]) -> float:
        """Calculate keyword overlap score (Enhancement 2)"""
        target_text = self._create_enhanced_embedding_text(target_row).lower()
        ref_text = self._create_enhanced_embedding_text(ref_data).lower() ##TRY
        # ref_text = self._create_reference_text(ref_data).lower()
        # logger.info(f"Rs-line345 enhanced_matching_engine.py - logging target_text: {target_text}")
        # logger.info(f"Rs-line346 enhanced_matching_engine.py - logging ref_text: {ref_text}")
        if not target_text or not ref_text:
            return 0.0
        
        # Extract meaningful keywords (remove common words)
        base_stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those'}
        own_stop_words = {'check', 'query', 'form', 'visit', 'desc', 'description',
       'logic', 'dataset', 'domain'}
        stop_words = base_stop_words.union(own_stop_words)
        target_words = set(word for word in re.findall(r'\b\w+\b', target_text) if len(word) > 2 and word not in stop_words)
        ref_words = set(word for word in re.findall(r'\b\w+\b', ref_text) if len(word) > 2 and word not in stop_words)
        
        if not target_words or not ref_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(target_words.intersection(ref_words))
        union = len(target_words.union(ref_words))
        # logger.info(f"Rs-line362 enhanced_matching_engine.py - logging intersection and union: {intersection},{union}")
        
        return intersection / union if union > 0 else 0.0
    
    def _create_reference_text(self, ref_data: Dict[str, Any]) -> str:
        """Create reference text for keyword comparison using standardized fields"""
        texts = []
        
        # Enhanced flexible field matching with regex/substring patterns
        field_patterns = [
            ['description', 'desc'],  # Matches any field containing 'description' or 'desc'
            ['query', 'text'],        # Matches any field containing 'query' or 'text'
            ['name', 'id'],           # Matches any field containing 'name' or 'id'
            ['form', 'dataset'],      # Matches any field containing 'form' or 'dataset'
            ['visit', 'timepoint'],   # Matches any field containing 'visit' or 'timepoint'
            ['code', 'logic', 'pseudo']  # Matches any field containing 'code', 'logic', or 'pseudo'
        ]
        
        # Dynamic field collection using substring matching
        for pattern_group in field_patterns:
            found_content = False
            for field_name, field_value in ref_data.items():
                if any(pattern.lower() in str(field_name).lower() for pattern in pattern_group):
                    if field_value and str(field_value).strip() != '' and str(field_value).strip().lower() not in ['n/a', 'na', '']:
                        texts.append(str(field_value))
                        found_content = True
                        break
            if found_content:
                continue  # Move to next pattern group
        
        return ' '.join(texts)
    
    
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
    
    def _is_meaningful_business_match(self, target_row: Dict[str, Any], ref_data: Dict[str, Any], score: float) -> bool:
        """Validate if match represents meaningful business logic alignment"""
        
        # Basic score threshold - primary filter
        if score < 0.15:  # Lowered from 0.25 to 0.15 (15%)
            return False
        
        # Extract key terms from both target and reference
        target_text = self._create_enhanced_embedding_text(target_row).lower()
        ref_text = self._create_reference_text(ref_data).lower()
        
        if not target_text or not ref_text:
            return False
        
        # Allow matches if either has sufficient content length (relaxed validation)
        if len(target_text) < 10 or len(ref_text) < 10:
            return False
        
        # Domain-specific validation patterns (optional check, not mandatory)
        lab_terms = {'lab', 'test', 'result', 'specimen', 'sample', 'normal', 'range', 'value'}
        ae_terms = {'adverse', 'event', 'ae', 'reaction', 'outcome', 'fatal', 'serious'}
        visit_terms = {'visit', 'date', 'completion', 'baseline', 'week', 'screening'}
        cm_terms = {'medication', 'concomitant', 'drug', 'treatment', 'therapy'}
        form_terms = {'form', 'ecrf', 'page', 'questionnaire', 'assessment'}
        
        domain_patterns = [lab_terms, ae_terms, visit_terms, cm_terms, form_terms]
        
        # Check for domain signals but don't require strict alignment
        target_domains = []
        ref_domains = []
        
        for i, pattern in enumerate(domain_patterns):
            target_matches = sum(1 for term in pattern if term in target_text)
            ref_matches = sum(1 for term in pattern if term in ref_text)
            
            if target_matches >= 1:  # IMPROVEMENT 1: Reduced required validation keyword count to 1 match instead of 2
                target_domains.append(i)
            if ref_matches >= 1:
                ref_domains.append(i)
        
        # IMPROVEMENT 2: Skip domain enforcement if hybrid score is very high (≥ 0.75)
        if score >= 0.75:
            return True  # Skip domain validation for very high scores
            
        # Normalize domain/dataset fields for fuzzy matching
        if target_domains and ref_domains:
            # Use fuzzy matching instead of exact domain comparison
            shared_domains = set(target_domains) & set(ref_domains)
            if not shared_domains and score < 0.40:
                # Check for partial domain overlap using normalized comparison
                target_normalized = [str(d)[:2] for d in target_domains]
                ref_normalized = [str(d)[:2] for d in ref_domains]
                has_normalized_overlap = bool(set(target_normalized) & set(ref_normalized))
                if not has_normalized_overlap:
                    return False
        
        # IMPROVEMENT 3: Add fallback logic - if cosine + keyword score ≥ 0.80, classify as "Excellent" regardless of domain match
        if score >= 0.80:
            return True  # Fallback logic for very high combined scores
        
        # Relaxed validation logic requirement
        validation_keywords = {'missing', 'present', 'required', 'mandatory', 'null', 'empty', 'check', 'verify', 'validate', 'ensure'}
        target_validation_count = sum(1 for term in validation_keywords if term in target_text)
        ref_validation_count = sum(1 for term in validation_keywords if term in ref_text)
        
        # Soften business logic filter - use looser validation
        if target_validation_count + ref_validation_count < 1:
            return False  # Looser filter
        
        return True  # More permissive validation
    
    def _generate_match_explanation(self, target_row: Dict[str, Any], ref_data: Dict[str, Any], 
                                   final_score: float, cosine_sim: float, keyword_overlap: float) -> str:
        """Generate GPT-based match explanation for traceability"""
        try:
            logger.info(f'Rs-line604: _generate_match_explanation called : {final_score},{cosine_sim},{keyword_overlap}')
            target_text = self._create_enhanced_embedding_text(target_row).lower()
            ref_text = self._create_enhanced_embedding_text(ref_data).lower() ##TRY
            # target_text = self._create_enhanced_embedding_text(target_row)
            # ref_text = self._create_reference_text(ref_data)
            
            if not target_text or not ref_text:
                return "Unable to generate explanation - insufficient text content"
            
            explanation_prompt = f"""
            Explain why these two MDD validation rules were matched by our semantic algorithm:

            TARGET RULE:
            {target_text}...

            REFERENCE RULE:
            {ref_text}...

            SCORES:
            - Final Score: {final_score:.3f}
            - Semantic Similarity: {cosine_sim:.3f} 
            - Keyword Overlap: {keyword_overlap:.3f}
            Provide Technically brief explanation of the key similarities that led to this match based on Lifescience in less than 100 words and Highlight primary reason of this match, also Highlight the difference for the Score reduction
            """
            # Provide Technically brief 2-3 sentence explanation of the key similarities that led to this match based on Lifescience in less than 200 letters also Include the Scores of the match            
            
            # Use GPT to generate explanation
            # explanation = self.client.generate_pseudo_code(explanation_prompt, "Match explanation request")
            extra_msg = (
                f"Individual scores: Final={final_score:.3f}, Semantic={cosine_sim:.3f}, Keywords={keyword_overlap:.3f}. "
                "Give a brief technical reason for each score."
            )
            explanation = self.client.generate_match_explanation(
                explanation_prompt,
                extra_user_message=extra_msg,
            )
            logger.info(f"Rs-line625: Generated explanation: {explanation}")
            # Fallback if GPT call fails
            if not explanation or len(explanation) < 20:
                return f"Match based on {final_score:.1%} similarity (semantic: {cosine_sim:.1%}, keywords: {keyword_overlap:.1%})"
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating match explanation: {str(e)}")
            return f"Match found with {final_score:.1%} confidence"

    # ------------------------------------------------------------------
    # New helper: generate technical (pseudo) logic when match is weak
    # ------------------------------------------------------------------
    def _generate_tech_logic(self, ref_data: Dict[str, Any]) -> str:
        """Generate SAS/SQL-style pseudo code for the reference rule.

        This is invoked when the overall match confidence is low (<30 %).
        It extracts the key descriptive elements from *ref_data* and delegates
        the heavy-lifting to AzureOpenAIClient.generate_pseudo_code().
        If the client is unavailable the function falls back to a static
        placeholder so that downstream code always receives a string.
        """
        try:
            # Extract core elements via canonical helpers first – fall back to raw keys (supports both ref_data and target_row)
            description = (
                _gf(ref_data, "dq_description")
                or ref_data.get("DQ Description")
                or ref_data.get("Target Check Description")
                or ""
            )
            query_text = (
                _gf(ref_data, "query_text")
                or ref_data.get("Query Text")
                or ref_data.get("Target Query Text")
                or ""
            )
            # Additional context for the model – helps it tailor the code
            form_name = _gf(ref_data, "P_form_name") or ref_data.get("Form Name") or ""
            visit_name = _gf(ref_data, "P_visit_name") or ref_data.get("Visit Name") or ""
            domain = _gf(ref_data, "primary_dataset") or ref_data.get("Domain") or ""
            variables = (
                _gf(ref_data, "primary_dataset_columns")
                or ref_data.get("Primary Domain Variables")
                or ""
            )
            data_category = _gf(ref_data, "data_category") or ref_data.get("Data Category: IDRP Data Category Name") or ""

            # Include Rule ID (from DQ Name) so TECH CODE can render like "AEAV151 = ..."
            rule_id = _gf(ref_data, "dq_name") or ref_data.get("DQ Name") or ""
            domain_context = {
                "domain": domain,
                "form": form_name,
                "variables": variables or "N/A",
                "visit": visit_name or "",
                "rule_id": rule_id or "",
                "data_category": data_category or ""
            }

            # If neither description nor query_text is present, nothing to do
            if not (description or query_text):
                return "// Pseudo-code generation skipped – insufficient reference data"

            if self.client and self.client.is_available():
                return (
                    self.client.generate_pseudo_code(
                        description,
                        query_text,
                        domain_context=domain_context,
                    )
                    or "// Pseudo-code generation returned empty string"
                )
            else:
                return (
                    "// Azure/OpenAI client unavailable – cannot auto-generate pseudo code.\n"
                    f"// Description: {description[:120]}..."
                )
        except Exception as exc:
            logger.error(f"_generate_tech_logic error: {exc}")
            return (
                "// Error generating pseudo code – see logs for details.\n"
                f"// Exception: {exc}"
            )
    
    def _generate_enhanced_match_explanation(self, target_row: Dict[str, Any], ref_data: Dict[str, Any], 
                                           final_score: float, cosine_sim: float, keyword_overlap: float) -> str:
        """Generate detailed match explanation with comprehensive business context"""
        try:
            # Extract key fields for detailed analysis
            target_desc = target_row.get('Target Check Description') or _gf(target_row, 'dq_description') or 'N/A'
            ref_desc = _gf(ref_data, 'dq_description') or ref_data.get('Reference Check Description', 'N/A')
            target_query = target_row.get('Target Query Text') or _gf(target_row, 'query_text') or 'N/A'
            ref_query = _gf(ref_data, 'query_text') or ref_data.get('Reference Query Text', 'N/A')
            target_form = _gf(target_row, 'form_name') or 'N/A'
            ref_form = _gf(ref_data, 'form_name') or 'N/A'
            ref_sponsor = _gf(ref_data, 'sponsor_name') or ref_data.get('Origin Study', 'Unknown')
            
            # Build comprehensive explanation
            explanation_parts = []
            
            # 1. Classification and confidence breakdown
            classification = self._classify_match_enhanced(final_score)
            explanation_parts.append(f"**{classification}** ({final_score:.1%} confidence)")
            
            # 2. Scoring methodology explanation
            explanation_parts.append(f"Hybrid Score Breakdown: {cosine_sim:.1%} semantic similarity + {keyword_overlap:.1%} keyword overlap")
            
            # 3. Semantic similarity analysis
            if cosine_sim >= 0.80:
                explanation_parts.append("✓ Very high semantic similarity - check logic and validation purpose are nearly identical")
            elif cosine_sim >= 0.60:
                explanation_parts.append("✓ Strong semantic similarity - validation concepts align well")
            elif cosine_sim >= 0.40:
                explanation_parts.append("✓ Moderate semantic similarity - related validation patterns detected")
            else:
                explanation_parts.append("⚠ Lower semantic similarity - validation logic differs but shows some alignment")
            
            # 4. Keyword overlap analysis
            if keyword_overlap >= 0.60:
                explanation_parts.append("✓ High keyword overlap - many shared clinical terms and data elements")
            elif keyword_overlap >= 0.40:
                explanation_parts.append("✓ Good keyword overlap - several matching clinical concepts")
            elif keyword_overlap >= 0.20:
                explanation_parts.append("✓ Some keyword overlap - contains related terminology")
            else:
                explanation_parts.append("⚠ Limited keyword overlap - different terminology but similar logic patterns")
            
            # 5. Domain and form analysis
            if target_form != 'N/A' and ref_form != 'N/A':
                if target_form.lower() == ref_form.lower():
                    explanation_parts.append(f"✓ Exact form match: Both apply to '{target_form}'")
                elif self._forms_are_similar(target_form, ref_form):
                    explanation_parts.append(f"✓ Similar forms: '{target_form}' ↔ '{ref_form}' (related data domains)")
                else:
                    explanation_parts.append(f"⚠ Different forms: '{target_form}' vs '{ref_form}' (cross-domain validation)")
            
            # 6. Query text similarity analysis
            if target_query != 'N/A' and ref_query != 'N/A':
                query_similarity = self._calculate_text_similarity(target_query.lower(), ref_query.lower())
                if query_similarity >= 0.60:
                    explanation_parts.append("✓ Very similar query messages - user experience would be consistent")
                elif query_similarity >= 0.30:
                    explanation_parts.append("✓ Related query messages - similar validation purpose")
                else:
                    explanation_parts.append("⚠ Different query messages but similar underlying validation logic")
            
            # 7. Reference source context
            explanation_parts.append(f"Reference source: {ref_sponsor} study validation library")
            
            # 8. Business recommendation
            if final_score >= 0.60:
                explanation_parts.append("💡 **Recommendation**: Excellent reuse candidate - high confidence match with proven validation logic")
            elif final_score >= 0.40:
                explanation_parts.append("💡 **Recommendation**: Good reuse candidate - requires minor customization")
            elif final_score >= 0.25:
                explanation_parts.append("💡 **Recommendation**: Consider for adaptation - similar validation pattern identified")
            elif final_score >= 0.15:
                explanation_parts.append("💡 **Recommendation**: Review for insights - may provide useful validation approach")
            else:
                explanation_parts.append("💡 **Recommendation**: Low confidence match - consider developing custom validation")
            
            # Join all parts with proper formatting
            return " | ".join(explanation_parts)
            
        except Exception as e:
            logger.error(f"Error generating enhanced match explanation: {str(e)}")
            return f"Match found with {final_score:.1%} confidence (semantic: {cosine_sim:.1%}, keywords: {keyword_overlap:.1%})"
    
    def _forms_are_similar(self, form1: str, form2: str) -> bool:
        """Check if two form names represent similar data domains"""
        form1_lower = form1.lower()
        form2_lower = form2.lower()
        
        # Define form similarity patterns
        lab_patterns = ['lab', 'laboratory', 'chemistry', 'hematology', 'urinalysis', 'specimen']
        ae_patterns = ['adverse', 'event', 'ae', 'reaction', 'outcome', 'fatal', 'serious']
        visit_patterns = ['visit', 'assessment', 'completion']
        cm_patterns = ['medication', 'concomitant', 'drug', 'treatment', 'therapy']
        pk_patterns = ['pharmacokinetic', 'pk', 'sampling']
        
        pattern_groups = [lab_patterns, ae_patterns, visit_patterns, cm_patterns, pk_patterns]
        
        for patterns in pattern_groups:
            form1_matches = any(pattern in form1_lower for pattern in patterns)
            form2_matches = any(pattern in form2_lower for pattern in patterns)
            if form1_matches and form2_matches:
                return True
        
        return False
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity using word overlap"""
        if not text1 or not text2:
            return 0.0
        
        # Tokenize and normalize
        words1 = set(word.lower().strip('.,!?') for word in text1.split() if len(word) > 2)
        words2 = set(word.lower().strip('.,!?') for word in text2.split() if len(word) > 2)
        
        # Remove common stop words
        stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'way', 'she', 'use', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye'}
        own_stop_words = {'check', 'query', 'form', 'visit', 'desc', 'description',
       'logic', 'dataset', 'domain'}
        stop_words = stop_words.union(own_stop_words)
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.index or not self.metadata:
            return {
                'total_vectors': 0,
                'vector_dimension': 0,
                'metadata_records': 0,
                'status': 'Not initialized'
            }
        
        return {
            'total_vectors': self.index.ntotal,
            'vector_dimension': self.index.d,
            'metadata_records': len(self.metadata),
            'status': 'Ready'
        }
    
    def is_initialized(self) -> bool:
        """Check if the matching engine is initialized"""
        return self.index is not None and len(self.metadata) > 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get matching statistics (alias for get_database_stats)"""
        return self.get_database_stats()
    
    def initialize_vector_db_with_files(self, database_files: List[str] = None) -> bool:
        """Initialize vector database with database files (compatibility method)"""
        # For compatibility with old matching engine calls
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
            self.embeddings = np.array(valid_embeddings).astype(np.float32)
            self.metadata = valid_metadata
            logger.info(f"Successfully rebuilt vector database with {len(valid_embeddings)} embeddings")
            ############RS FAISS 
            # Save FAISS index
            import faiss
            # self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            
            # Replace IndexFlatL2 with IndexFlatIP (Inner Product) and normalize vectors
            self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
            faiss.normalize_L2(self.embeddings)  # Normalize the vectors
            self.index.add(self.embeddings)
            # Save FAISS index
            # Write index locally and upload if needed
            index_path = 'data/faiss_index.bin'
            if USE_S3:
                tmp_dir = pathlib.Path('/tmp/data'); tmp_dir.mkdir(parents=True, exist_ok=True)
                index_path = str(tmp_dir / 'faiss_index.bin')
            faiss.write_index(self.index, index_path)
            if USE_S3:
                with open(index_path, 'rb') as _f:
                    storage.write_bytes('data/faiss_index.bin', _f.read())
            logger.info(f"Successfully rebuilt FAISS index with {len(valid_embeddings)} embeddings")
            ##############

            
            # Build sponsor index map for same-sponsor queries in this process
            try:
                self._build_sponsor_index_map()
            except Exception as _e:
                logger.warning(f"Failed to build sponsor index map after rebuild: {str(_e)}")

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

    def append_vector_database(self, mdd_files: List[str], progress_cb: Optional[Callable[[int, int, str], None]] = None) -> bool:
        """Append new vectors from the supplied MDD files to the existing vector database.
        It loads the existing index/embeddings if necessary and only processes files
        that were not part of the database previously (identified via `source_file`)."""
        try:
            logger = logging.getLogger(__name__)

            # Ensure the current database is loaded (if it exists)
            logger.info(f"Before loading: is_initialized={self.is_initialized()}, metadata count={len(self.metadata) if self.metadata else 0}")
            
            # Always load the latest metadata to ensure we have the most recent data
            try:
                # Check if metadata files exist before trying to load them
                metadata_path = os.path.join('data', 'metadata.pkl')
                exists_meta = storage.exists('data/metadata.pkl') if USE_S3 else os.path.exists(metadata_path)
                if exists_meta:
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
                # Report immediate completion if progress callback is provided
                if progress_cb is not None:
                    try:
                        progress_cb(1, 1, "No new files to process")
                    except Exception:
                        pass
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
                    # Prefer field-alias resolution; fallback to direct key if needed
                    val = _gf(rec, key)
                    if val is None or str(val).strip() == '':
                        val = rec.get(key)
                    if val is None or str(val).strip() == '':
                        continue
                    signature_parts.append(f"{key}:{_normalise_value(val)}")
                
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
                # logger.info(f"Rs-line854 enhanced_matching_engine.py append_vector_database - logging sig: {sig}")
                # logger.info(f"Rs-line855 enhanced_matching_engine.py append_vector_database - logging existing_keys: {existing_keys}")
                # logger.info(f"Rs-line856 enhanced_matching_engine.py append_vector_database - logging batch_keys: {batch_keys}")
                # logger.info(f"Rs-line857 enhanced_matching_engine.py append_vector_database - logging sig in existing_keys: {sig in existing_keys}")
                # logger.info(f"Rs-line858 enhanced_matching_engine.py append_vector_database - logging sig in batch_keys: {sig in batch_keys}")
                
                if sig == "":
                    logger.info(f"Rs-line859 enhanced_matching_engine.py append_vector_database - empty signature detected")
                    continue
                if sig in existing_keys or sig in batch_keys:
                    logger.info(f"Rs-line858 enhanced_matching_engine.py append_vector_database - duplicate detected")
                    continue  # duplicate detected
                unique_records.append(rec)
                batch_keys.add(sig)

            if not unique_records:
                logger.info("All records in the new files are duplicates; nothing to append")
                if progress_cb is not None:
                    try:
                        progress_cb(1, 1, "No unique records after de-duplication")
                    except Exception:
                        pass
                return True

            # Generate embeddings only for unique records
            # logger.info(f"Rs-line866 enhanced_matching_engine.py append_vector_database - logging unique_records: {unique_records}")
            embedding_texts = [self._create_enhanced_embedding_text(rec) for rec in unique_records]
            total_to_embed = len(embedding_texts)
            processed = 0

            def on_embed_progress():
                nonlocal processed, total_to_embed
                processed += 1
                if progress_cb is not None:
                    try:
                        progress_cb(processed, total_to_embed, f"Generating embeddings {processed}/{total_to_embed}")
                    except Exception:
                        pass

            # Initial progress notification before starting embeddings
            if progress_cb is not None:
                try:
                    progress_cb(0, total_to_embed, f"Starting embedding generation for {total_to_embed} records")
                except Exception:
                    pass

            embeddings = self.client.get_embeddings_batch(embedding_texts, on_progress=on_embed_progress)
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
            # Rebuild sponsor index map to include new records
            try:
                self._build_sponsor_index_map()
            except Exception as _e:
                logger.warning(f"Failed to update sponsor index map after append: {str(_e)}")
            
            # Persist updated database
            index_path = 'data/faiss_index.bin'
            if USE_S3:
                tmp_dir = pathlib.Path('/tmp/data'); tmp_dir.mkdir(parents=True, exist_ok=True)
                index_path = str(tmp_dir / 'faiss_index.bin')
            faiss.write_index(self.index, index_path)
            if USE_S3:
                with open(index_path, 'rb') as _f:
                    storage.write_bytes('data/faiss_index.bin', _f.read())
            # Finalizing persistence
            self._save_precomputed_embeddings()
            if progress_cb is not None:
                try:
                    progress_cb(total_to_embed, total_to_embed, "Finalizing and saving database")
                except Exception:
                    pass
            logger.info(f"Appended {len(valid_embeddings)} new embeddings. Total vectors: {self.index.ntotal}")
            return True
        except Exception as e:
            logger.error(f"Error appending to vector database: {str(e)}")
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
            tmp_dir = pathlib.Path('/tmp/data') if USE_S3 else None
            if USE_S3 and tmp_dir is not None:
                tmp_dir.mkdir(parents=True, exist_ok=True)
            
            # Save embeddings
            embeddings_path = os.path.join(data_dir, 'embeddings.npy')
            if USE_S3 and tmp_dir is not None:
                # write to tmp then upload
                emb_tmp = str(tmp_dir / 'embeddings.npy')
                np.save(emb_tmp, self.embeddings)
                with open(emb_tmp, 'rb') as _f:
                    storage.write_bytes('data/embeddings.npy', _f.read())
            else:
                np.save(embeddings_path, self.embeddings)
            
            # Save metadata
            metadata_path = os.path.join(data_dir, 'metadata.pkl')
            if USE_S3:
                import pickle as _pickle
                storage.write_bytes('data/metadata.pkl', _pickle.dumps(self.metadata))
            else:
                with open(metadata_path, 'wb') as f:
                    pickle.dump(self.metadata, f)
            
            # Save embeddings
            embeddings_path = os.path.join(data_dir, 'precomputed_embeddings.npy')
            if USE_S3 and tmp_dir is not None:
                pre_tmp = str(tmp_dir / 'precomputed_embeddings.npy')
                np.save(pre_tmp, self.embeddings)
                with open(pre_tmp, 'rb') as _f:
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