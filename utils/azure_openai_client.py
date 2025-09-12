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
