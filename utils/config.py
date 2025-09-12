import os
import logging
from typing import Iterable, Optional

logger = logging.getLogger(__name__)
# print(logger)
# logger.info(f'Rs-config.py line 6')
def get_secret_details(prod_account):
    import boto3
    import traceback
    from botocore.exceptions import ClientError
    import json
    try:
        session = boto3.Session()
        sm_client = session.client(
            service_name="secretsmanager",
            region_name="us-east-1"
            )
        print('client_det', sm_client)
        secret_id_value = 'mdd/generate/utilities'
        # #fetch master secret 
        # try:
        #     get_master_secret_value_response = sm_client.get_secret_value(
        #             SecretId="account_promotion_master_secret"
        #         )
        # except ClientError as e:
        #     raise RuntimeError(f"Unable to retrieve secret: {e}")
        # master_secret_string = get_master_secret_value_response.get('SecretString')
        # if master_secret_string:
        #     master_secret_data = json.loads(master_secret_string)
        #     if prod_account in master_secret_data:
        #         secret_id_value = master_secret_data[prod_account]
        try:
            get_secret_value_response = sm_client.get_secret_value(
                    SecretId=secret_id_value
                )
        except ClientError as e:
            raise RuntimeError(f"Unable to retrieve secret: {e}")
        secret_string = get_secret_value_response.get('SecretString')
        if secret_string:
            return json.loads(secret_string)
        else:
            return False
    except Exception as error:
        print(f"Error while connecting to AWS Secret Manager - {str(error)}")
        return False

def _truthy(val: Optional[str]) -> bool:
    if val is None:
        return False
    return val.strip().lower() in {"1", "true", "yes", "on"}

# Decide whether to load .env based solely on USE_DOTENV
try:
    from pathlib import Path
    _has_env_file = Path(".env").exists()
except Exception:
    _has_env_file = False
_use_dotenv_flag = os.getenv("USE_DOTENV")
logger.info(f'Rs-config.py line 19 _use_dotenv_flag {_use_dotenv_flag}')
logger.info(f'Rs-config.py line 20 _has_env_file {_has_env_file}')
if _use_dotenv_flag is None and _has_env_file:
    try:
        from dotenv import dotenv_values  # type: ignore
        _use_dotenv_flag = dotenv_values(".env").get("USE_DOTENV")
    except Exception:
        _use_dotenv_flag = None

_USE_DOTENV = _truthy(_use_dotenv_flag) and _has_env_file
logger.info(f'Rs-config.py line 26 _USE_DOTENV {_USE_DOTENV}')
if _USE_DOTENV:
    try:
        from dotenv import load_dotenv  # type: ignore
        load_dotenv()
        logger.info("Loaded .env for local development")
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not load .env (this is fine if variables are set another way): {e}")


# --- AWS Secrets Manager Integration for Production ---
_PROD_SECRETS = None
if not _USE_DOTENV:
    try:
        # You may want to customize prod_account as needed
        _PROD_SECRETS = get_secret_details(prod_account="production")
        if not _PROD_SECRETS:
            logger.warning("No secrets loaded from AWS Secrets Manager.")
    except Exception as e:
        logger.error(f"Failed to load secrets from AWS Secrets Manager: {e}")


def env_str(name: str, default: Optional[str] = None, required: bool = False, alt_names: Optional[Iterable[str]] = None) -> Optional[str]:
    """Read a string environment variable or AWS secret in production.

    - alt_names: optional list of alternative names to check if 'name' is not set.
    - if required=True and value not found, raises RuntimeError.
    """
    val = None
    logger.info(f'Rs-config.py line 99 name {name}')
    
    if not _USE_DOTENV and _PROD_SECRETS:
        val = _PROD_SECRETS.get(name)
        logger.info(f'Rs-config.py line 101 val {val}')
        if not val and alt_names:
            for alt in alt_names:
                val = _PROD_SECRETS.get(alt)
                logger.info(f'Rs-config.py line 103 alt {alt}')
                if val:
                    break
    else:
        val = os.getenv(name)
        logger.info(f'Rs-config.py line 106 val {val}')
        if not val and alt_names:
            for alt in alt_names:
                val = os.getenv(alt)
                logger.info(f'Rs-config.py line 108 alt {alt}')
                if val:
                    break
    if val:
        logger.info(f'Rs-config.py line 110 val {val}')
        return val
    if default is not None:
        return default
    if required:
        names = ", ".join([name] + list(alt_names or []))
        raise RuntimeError(f"Missing required config variable(s): {names}")
    return None


def env_bool(name: str, default: bool = False) -> bool:
    val = None
    if not _USE_DOTENV and _PROD_SECRETS:
        val = _PROD_SECRETS.get(name)
    else:
        val = os.getenv(name)
    return _truthy(val) if val is not None else default


def env_int(name: str, default: int) -> int:
    val = None
    if not _USE_DOTENV and _PROD_SECRETS:
        val = _PROD_SECRETS.get(name)
    else:
        val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        raise RuntimeError(f"Config variable {name} must be an integer, got: {val}")


IS_DEV = _USE_DOTENV
logger.info(f'Rs-config.py line 81 IS_DEV {IS_DEV}')
# Common app settings
_DEFAULT_DEV_SECRET = "mdd-automation-secret-key-2024"
SESSION_SECRET: str = env_str(
    "SESSION_SECRET",
    default=_DEFAULT_DEV_SECRET if IS_DEV else None,
    required=not IS_DEV,
) or _DEFAULT_DEV_SECRET  # fallback only for type-checkers; logic above ensures prod requires it

DEBUG: bool = env_bool("DEBUG", default=IS_DEV)
PORT: int = env_int("PORT", default=5002)

# Storage settings
USE_S3: bool = env_bool("USE_S3", default=False)
S3_BUCKET: Optional[str] = env_str("S3_BUCKET") if USE_S3 else None
logger.info(f'Rs-config.py line 127 S3_BUCKET {S3_BUCKET} USE_S3 : {USE_S3}')
def validate_config() -> None:
    """Fail fast in production if required settings are missing."""
    if not IS_DEV and not SESSION_SECRET:
        raise RuntimeError("SESSION_SECRET must be set in production.")
    if not IS_DEV and USE_S3 and not S3_BUCKET:
        raise RuntimeError("S3_BUCKET must be set in production if USE_S3 is True.")


__all__ = [
    "env_str",
    "env_bool",
    "env_int",
    "SESSION_SECRET",
    "DEBUG",
    "PORT",
    "IS_DEV",
    "validate_config",
    "USE_S3",
    "S3_BUCKET",
]

''' For setting environment variables in Linux/MacOS
# Flask
# export SESSION_SECRET="your-strong-secret"

# Azure / OpenAI
# export AZURE_OPENAI_API_KEY="..."
# export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
# export AZURE_OPENAI_API_VERSION="2024-06-01"
# (or the non-Azure OPENAI_* equivalents)
'''
