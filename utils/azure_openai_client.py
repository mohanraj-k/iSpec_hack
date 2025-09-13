"""
Azure OpenAI Client for embeddings and pseudo code generation
"""

import os
import json
import logging
from typing import List, Optional, Dict, Any, Callable
# from openai import OpenAI
from openai import AzureOpenAI
from .config import env_str

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """Client for OpenAI API operations (keeping class name for compatibility)"""
    
    def __init__(self):
        """Initialize OpenAI client"""
        # Allow overriding deployment/model names via environment variables
        self.embedding_model = env_str(
            "OPENAI_EMBEDDING_MODEL",
            alt_names=["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
            default="text-embedding-3-small",
        ) or "text-embedding-3-small"
        self.chat_model = env_str(
            "OPENAI_CHAT_MODEL",
            alt_names=["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            default="gpt-4o",
        ) or "gpt-4o"
        
        # Get OpenAI API key
        api_key = env_str("OPENAI_API_KEY", alt_names=["AZURE_OPENAI_API_KEY"])  # Azure/OpenAI compatibility
        api_version = env_str("OPENAI_API_VERSION", alt_names=["AZURE_OPENAI_API_VERSION"]) or "2024-06-01"
        endpoint = env_str("OPENAI_API_BASE", alt_names=["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_BASE"])  # Azure SDK expects azure_endpoint
        # logger.info(f'Rs-OpenAI API key: {api_key}')
        # logger.info(f'Rs-OpenAI API version: {api_version}')
        # logger.info(f'Rs-OpenAI API endpoint: {endpoint}')
        if not api_key or not endpoint:
            logger.info(f'Rs-line24')
            if not api_key:
                logger.error("OpenAI API key not found in environment")
            if not endpoint:
                logger.error("OpenAI/Azure endpoint not found in environment (OPENAI_API_BASE or AZURE_OPENAI_ENDPOINT)")
            self.client = None
        else:
            try:
                # self.client = OpenAI(
                #     api_key=api_key,
                #     timeout=30.0,
                #     max_retries=2
                # )
                self.client = AzureOpenAI(
                    api_key=api_key,
                    timeout=30.0,
                    max_retries=2,
                    api_version=api_version,
                    azure_endpoint=endpoint
                )
                logger.info("OpenAI client initialized successfully")

            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                self.client = None
    
    def get_embedding(self, text: str, max_retries: int = 3) -> Optional[List[float]]:
        """Get embedding vector for text with retry logic"""
        if not self.client:
            logger.warning("Azure OpenAI client not available, returning None")
            return None
        
        # Clean and prepare text
        text = self._clean_text(text)
        if not text or not text.strip():
            return None
        
        # Truncate text if too long
        if len(text) > 4000:  # Reduce to prevent timeouts
            text = text[:4000]
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=text,
                    model=self.embedding_model,
                    timeout=15  # Shorter timeout per request
                )
                
                if response.data and len(response.data) > 0:
                    embedding = response.data[0].embedding
                    if attempt > 0:
                        logger.info(f"Successfully generated embedding on attempt {attempt + 1}")
                    return embedding
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"All embedding attempts failed")
                    return None
                
                # Wait before retry with exponential backoff
                import time
                time.sleep(1 + attempt)
        
        return None
    
    def get_embeddings_batch(self, texts: List[str], on_progress: Optional[Callable[[], None]] = None) -> List[Optional[List[float]]]:
        """Get embeddings for multiple texts.
        If provided, on_progress() will be called after each text is processed.
        """
        if not self.client:
            logger.warning("Azure OpenAI client not available")
            return [None] * len(texts)
        
        embeddings = []
        batch_size = 100  # Process in batches to avoid rate limits
        logger.info(f'Rs-line101 - logging texts: {len(texts)}') ## Entire Library MDD rows input
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                embedding = self.get_embedding(text)
                batch_embeddings.append(embedding)
                if on_progress is not None:
                    try:
                        on_progress()
                    except Exception:
                        # Progress callbacks should never break the main flow
                        pass
            logger.info(f'Rs-line110 - logging batch_embeddings: {len(batch_embeddings)}')
            embeddings.extend(batch_embeddings)
            logger.info(f'Rs-line112 - logging embeddings: {len(embeddings)}')
            logger.debug(f"Processed embedding batch {i//batch_size + 1}")
        
        return embeddings
    
    def generate_pseudo_code(self, check_description: str, query_text: str, 
                           domain_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate pseudo code for unmatched data quality checks"""
        if not self.client:
            logger.warning("Azure OpenAI client not available, returning default pseudo code")
            return f"// Auto-generated pseudo code for: {check_description[:50]}..."
        
        try:
            # Skip pseudo code generation to prevent timeouts during upload processing
            logger.info("Skipping pseudo code generation to prevent timeout issues")
            # return f"// Pseudo code generation skipped to ensure fast processing\n// Check: {check_description[:100]}..."
            # Prepare context
            context_info = ""
            if domain_context:
                context_info = f"""
            Domain: {domain_context.get('domain', 'Unknown')}
            Form: {domain_context.get('form', 'Unknown')}
            Variables: {domain_context.get('variables', 'Unknown')}
            Visit: {domain_context.get('visit', 'Unknown')}
            Rule ID: {domain_context.get('rule_id', 'Unknown')}
            Data Category: {domain_context.get('data_category', 'Unknown')}
            """
                        
            prompt = f"""
            You are a clinical data programming expert and with Knowledge of RAVE and VIVA edc and with Based on the details below, produce 1 output:

            INPUTS
            Check Description: {check_description}
            Query Text: {query_text}
            {context_info}

            OUTPUT FORMAT (return exactly these sections, no extra commentary, no markdown fences):
            TECH CODE:
            - A single line boolean rule using the available variables.
            - Example style: AETERM = FEVER AND ECSTDTC > AEENDTC. Query should fire
            - Use UPPERCASE for variables; keep it concise and deterministic.

            """
            #PSEUDO CODE:
            # - A compact SAS/SQL-style pseudo implementation.
            # - Follow clinical data programming best practices and handle missing data.
            # - Keep it short (<= 30 lines), clear, and directly implement the TECH CODE logic.
            # - Do not include explanations outside of code comments.
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a clinical data programming expert specializing in data quality checks."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            pseudo_code = response.choices[0].message.content.strip()
            logger.debug("Generated pseudo code successfully")
            logger.info(f'Rs-line187 - logging pseudo_code: {pseudo_code}')
            return pseudo_code
            
        except Exception as e:
            logger.error(f"Error generating pseudo code: {str(e)}")
            return f"// Error generating pseudo code: {str(e)}\n// Manual review required for: {check_description}"
    
    def generate_match_explanation(self, explanation_prompt: str, *, extra_user_message: Optional[str] = None, max_tokens: int = 180, temperature: float = 0.2) -> str:
        """Generate a concise match explanation using the chat model.

        Parameters:
            explanation_prompt: The fully constructed prompt describing target and reference rules and scores.
            max_tokens: Maximum tokens for the response to keep it brief.
            temperature: Sampling temperature for controllable determinism.

        Returns:
            A short explanation (aim for <= 200 characters) or a fallback message if generation is unavailable.
        """
        if not self.client:
            logger.warning("Azure OpenAI client not available, returning default match explanation")
            return (
                "Automated explanation unavailable. Match is based on semantic and keyword similarity; review aligned terms and domains."
            )

        try:
            system_msg = (
                "You are a clinical data programming expert. Provide very concise, plain-language rationales for why two data "
                "validation rules match. Keep responses under 100 words, also Display the Scores which are passed in the prompt "
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": explanation_prompt},
            ]
            if extra_user_message:
                messages.append({"role": "user", "content": extra_user_message})

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            logger.info(f'Rs-line223 enhanced_matching_engine.py - logging explanation_prompt: {explanation_prompt}')
            explanation = (response.choices[0].message.content or "").strip()
            if not explanation:
                return (
                    "Explanation not generated. Match reflects similar variables/conditions with strong semantic and keyword overlap."
                )
            # Ensure brevity
            return explanation
        except Exception as e:
            logger.error(f"Error generating match explanation: {str(e)}")
            return (
                f"Error during explanation generation: {str(e)}. Match likely due to similar rule intent and terminology."
            )
    
    def adapt_pseudo_code(self, base_pseudo_code: str, target_context: Dict[str, Any],
                         reference_context: Dict[str, Any]) -> str:
        """Adapt pseudo code from reference to target context"""
        if not self.client:
            logger.warning("Azure OpenAI client not available, returning base pseudo code")
            return base_pseudo_code
        
        try:
            # Skip pseudo code adaptation to prevent timeouts during upload processing  
            logger.info("Skipping pseudo code adaptation to prevent timeout issues")
            return f"// Pseudo code adaptation skipped to ensure fast processing\n// Base: {base_pseudo_code[:100]}..."
            prompt = f"""
Adapt the following pseudo code from a reference context to a new target context:

Original Pseudo Code:
{base_pseudo_code}

Reference Context:
Domain: {reference_context.get('domain', 'Unknown')}
Form: {reference_context.get('form', 'Unknown')}
Variables: {reference_context.get('variables', 'Unknown')}

Target Context:
Domain: {target_context.get('domain', 'Unknown')}
Form: {target_context.get('form', 'Unknown')}
Variables: {target_context.get('variables', 'Unknown')}

Please adapt the pseudo code to work with the target context while maintaining the same logical intent. Make necessary variable name changes and add comments explaining the adaptations.

Adapted Pseudo Code:
"""
            
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a clinical data programming expert."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            adapted_code = response.choices[0].message.content.strip()
            logger.debug("Adapted pseudo code successfully")
            return adapted_code
            
        except Exception as e:
            logger.error(f"Error adapting pseudo code: {str(e)}")
            return f"{base_pseudo_code}\n\n// Adaptation error: {str(e)}\n// Manual review required"
    
    def generate_python_script(self, dq_name: str, check_description: str = "", query_text: str = "", tech_code: str = "", domain_context: Optional[Dict[str, Any]] = None) -> str:
        """Generate a runnable Python DQ script from TECH CODE and context.

        Output requirements (Template-style):
        - Import pandas as pd (no other third-party imports).
        - Define a single function named: rulefn_<sanitized DQ Name>
          Signature:
            def rulefn_<name>(query_text: str,
                              primary_df: pd.DataFrame,
                              r1_df: pd.DataFrame | None = None,
                              r2_df: pd.DataFrame | None = None,
                              r3_df: pd.DataFrame | None = None,
                              r4_df: pd.DataFrame | None = None,
                              r5_df: pd.DataFrame | None = None) -> list[dict]
        - Each Domain is a DataFrame; each Variable is a column in that domain's DataFrame.
        - Always read primary variables from primary_df[...] and relational variables from the respective rN_df[...]. Do not interchange.
        - Return a list[dict] payload_records where each payload has keys:
            {"query_text": str, "form_index": str, "modif_dts": str, "stg_ck_event_id": int, "relational_ck_event_ids": list, "confid_score": int}
        - Keep module self-contained (no undefined helper functions). If you need helpers, define them above the function.
        """
        # Prepare a safe function name token based on dq_name
        safe_base = (dq_name or "DQ_Script").strip() or "DQ_Script"
        fn_token = "".join(ch if ch.isalnum() else "_" for ch in safe_base)
        if not fn_token:
            fn_token = "DQ_Script"
        if fn_token[0].isdigit():
            fn_token = "X_" + fn_token
        fn_name = f"rulefn_{fn_token}"
        # Build dynamic, domain-named parameters for the function signature
        ctx0 = domain_context or {}
        def _san(n: Any, default: str) -> str:
            s = str(n or "").strip()
            if not s:
                return default
            # Keep alnum/underscore, collapse others to underscore, uppercase like template style
            s2 = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)
            s2 = s2.strip("_") or default
            if s2 and s2[0].isdigit():
                s2 = f"X_{s2}"
            return s2.upper()

        primary_param = _san(ctx0.get('domain'), 'PRIMARY')
        _rels_raw = ctx0.get('relational_domains', '')
        if isinstance(_rels_raw, (list, tuple)):
            _raw_list = [str(x).strip() for x in _rels_raw if str(x).strip()]
        else:
            _raw_list = [x.strip() for x in str(_rels_raw).split(',') if x.strip()]
        rel_params: List[str] = []
        for _nm in _raw_list[:5]:
            _s = _san(_nm, '')
            if _s and _s not in rel_params:
                rel_params.append(_s)
        # Compose function signature string and domains line for docstring
        function_sig = f"def {fn_name}(query_text: str, {primary_param}: pd.DataFrame"
        for _rp in rel_params:
            function_sig += f", {_rp}: pd.DataFrame | None = None"
        function_sig += ") -> list[dict]"
        domains_line = ", ".join([primary_param] + rel_params) if rel_params else primary_param
        params_hint_line = ", ".join([primary_param] + rel_params) if rel_params else primary_param
        # Fallback stub if client unavailable
        if not self.client:
            safe_name = (dq_name or "DQ_Script").strip() or "DQ_Script"
            stub = f'''"""
DQ Script: {safe_name}
Auto-generated stub (OpenAI unavailable).
Tech Code (for reference): {tech_code or 'N/A'}
"""

from typing import List, Dict, Any, Optional
import pandas as pd


{function_sig}:
    """Template-style DQ function returning a list of payload dicts.

    Each Domain is a DataFrame; each Variable is a column in that domain's DataFrame.
    Always read primary variables from {primary_param}[...] and relational variables from the respective domain DataFrames. Do not interchange.
    """
    try:
        payload_records: List[Dict[str, Any]] = []
        # TODO: Implement rule from TECH CODE
        return payload_records
    except Exception as e:
        # Return empty payloads on error to avoid breaking pipelines
        return []
'''
            return stub

        # Construct prompt for deterministic, dependency-free Python
        try:
            ctx = domain_context or {}
            ctx_str = (
                f"Domain: {ctx.get('domain', 'Unknown')}\n"
                f"Form: {ctx.get('form', 'Unknown')}\n"
                f"Visit: {ctx.get('visit', 'Unknown')}\n"
                f"Primary Variables: {ctx.get('variables', 'Unknown')}\n"
                f"Relational Domains: {ctx.get('relational_domains', '')}\n"
                f"Relational Variables: {ctx.get('relational_variables', '')}\n"
                f"Relational Dynamic Variables: {ctx.get('relational_dynamic_variables', '')}\n"
            )

            prompt = (
                "Act as a senior clinical data programming engineer.\n"
                "Write a single self-contained Python module that implements a data quality rule following this template style.\n"
                "Rules:\n"
                "- Output ONLY valid Python code (no markdown, no fences).\n"
                "- You may import pandas as pd. Do NOT import any other third-party packages.\n"
                "- Define exactly one function with this signature and name:\n"
                f"  {function_sig}\n"
                f"- Use these DataFrame parameter names exactly: {params_hint_line}.\n"
                "- Include a short header docstring with:\n"
                f"  Rule Name: {fn_name} ##DQName\n"
                f"  Domains: {domains_line} ##Domains\n"
                "  ## Each variable is a column to the DataFrame of respective domain.\n"
                "- Each Domain is a DataFrame; each Variable is a column in that domain.\n"
                f"  Always read primary variables from {primary_param}[...] and relational variables from the respective domain DataFrames. Do not interchange.\n"
                "- Return a list[dict] named payload_records. Each payload dict must exactly contain keys:\n"
                "  'query_text' (str), 'form_index' (str), 'modif_dts' (str), 'stg_ck_event_id' (int), 'relational_ck_event_ids' (list), 'confid_score' (int).\n"
                "- Handle missing columns/values defensively: use .get-style accessors or existence checks; do not crash.\n"
                "- Keep code concise, readable, deterministic, and self-contained (define any helpers you use inside this file).\n\n"
                f"DQ Name: {dq_name}\n"
                f"TECH CODE: {tech_code}\n"
                f"CHECK DESCRIPTION: {check_description}\n"
                f"QUERY TEXT: {query_text}\n"
                f"CONTEXT:\n{ctx_str}\n"
                "Return only the Python file content implementing the rule."
            )

            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You turn rule specifications into robust, dependency-free Python functions."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1100,
                temperature=0.2,
            )
            content = (response.choices[0].message.content or "").strip()
            # Basic guard: ensure function name exists
            if f"def {fn_name}(" not in content:
                content += (
                    "\n\n\n# Fallback: ensure expected entrypoint exists\n"
                    "from typing import List, Dict, Any, Optional\n"
                    "import pandas as pd\n\n"
                    f"{function_sig}:\n"
                    "    return []\n"
                )
            return content
        except Exception as e:
            logger.error(f"Error generating Python script: {str(e)}")
            safe_name = (dq_name or "DQ_Script").strip() or "DQ_Script"
            return f'''"""
DQ Script: {safe_name}
Generation error fallback. See server logs.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


{function_sig}:
    try:
        return []
    except Exception:
        return []
'''

    def _clean_text(self, text: str) -> str:
        """Clean text for embedding"""
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Remove excessive whitespace and normalize
        text = " ".join(text.split())
        
        # Limit length to avoid token limits
        max_length = 8000  # Conservative limit for embeddings
        if len(text) > max_length:
            text = text[:max_length]
            logger.debug(f"Truncated text to {max_length} characters")
        
        return text
    
    def clean_logic_text(self, text: str) -> str:
        """Enhanced text cleaning for better embedding accuracy"""
        import re
        
        if not text:
            return ""
        
        text = str(text).lower().strip()
        
        # Remove common placeholder text that adds noise
        text = text.replace("query should fire", "")
        text = text.replace("please update", "")
        text = text.replace("dm review:", "")
        
        # Remove placeholders like [%VARIABLE%]
        text = re.sub(r'\[%.*?\%\]', '', text)
        
        return text
    
    def is_available(self) -> bool:
        """Check if Azure OpenAI client is available"""
        return self.client is not None

    # ----------------- Label->Variable Mapping Helper -----------------
    def suggest_label_to_var_mapping(self, code: str, candidates: List[Dict[str, str]], domain: str = "") -> Dict[str, str]:
        """Use the chat model to suggest a mapping from variable label phrases (human-friendly) to
        variable names (dataset-friendly) found or logically implied in the provided code.

        Parameters:
            code: The TECH CODE / PSEUDO CODE text containing label-like phrases.
            candidates: List of {"label": <label>, "name": <var_name>} candidate pairs from the CSV.
            domain: Optional domain context (e.g., AE, LB) to bias mapping.

        Returns:
            Dict mapping label strings (exact label from candidates) -> variable name strings.
        """
        try:
            if not self.client:
                return {}
            if not code or not candidates:
                return {}

            # Prepare compact candidate list to keep prompt small
            # Format: label|name per line
            cand_lines = []
            for c in candidates[:100]:
                lab = (c.get("label") or "").strip()
                nm = (c.get("name") or "").strip()
                if lab and nm:
                    cand_lines.append(f"{lab} | {nm}")
            cand_block = "\n".join(cand_lines)

            system_msg = (
                "You are a clinical data programming expert. Map human-friendly variable labels to"
                " dataset variable names used in clinical EDC/SDTM-like contexts."
            )
            user_prompt = (
                f"Domain: {domain or 'Unknown'}\n"
                f"CODE:\n{code}\n\n"
                "CANDIDATE LABELS (format: label | variable_name):\n"
                f"{cand_block}\n\n"
                "Task: Identify which candidate labels are present (or logically equivalent) in the CODE and"
                " return a JSON object that maps label (exact candidate label string) to the variable_name.\n"
                "- Consider partial/semantic matches (e.g., 'Adverse Event Term' -> AETERM).\n"
                "- Prefer domain-appropriate choices.\n"
                "- Only include labels that you are confident apply to the CODE.\n"
                "Output JSON only, no commentary. Example: {\"Adverse Event Term\": \"AETERM\"}"
            )

            resp = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=400,
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
            # Parse JSON strictly, with fallback to substring extraction
            try:
                return json.loads(content)
            except Exception:
                # Attempt to locate JSON braces
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    sub = content[start:end+1]
                    try:
                        return json.loads(sub)
                    except Exception:
                        return {}
                return {}
        except Exception as e:
            logger.warning(f"suggest_label_to_var_mapping failed: {e}")
            return {}
