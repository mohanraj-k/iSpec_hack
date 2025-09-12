"""Central registry for all column/field name variations used in MDD automation. Each canonical key maps to list of aliases."""
from typing import Any, Dict, List

FIELD_ALIASES: Dict[str, List[str]] = {
    # Core identifiers
    "dq_name": ["DQ Name", "DQ name", "Check Name", "Plan Name", "name", "dq_name","Check name"],
    "dq_description": ["DQ Description", "DQ description", "Edit Check Description", "Check Description", "DESCRIPTION", "Description", "dq_description"],
    "query_text": ["Standard Query text", "Standard Query Text", "Query Text", "Query", "query", "query_text"],
    "pseudo_code": ["Pseudo Code", "pseudo code", "Check logic", "Check Logic", "logic", "pseudo_code"],
    # Study / sponsor
    "sponsor_name": ["Sponsor", "Sponsor Name", "sponsor_name","source_file"],
    "study_id": ["Study", "Study ID", "Source Study ID", "study_id"],
    # Dataset / domain
    "primary_dataset": ["Primary Dataset", "Primary Domain", "Data Domain", "Domain", "domain", "primary_dataset","Primary Dynamic Domain"],
    "R1_Domain": ["relational_dataset_1","Relational Dataset_1","Relational Dataset"],
    "R2_Domain": ["relational_dataset_2","Relational Dataset_2"],
    "R3_Domain": ["relational_dataset_3","Relational Dataset_3"],
    "R4_Domain": ["relational_dataset_4","Relational Dataset_4"],
    "R5_Domain": ["relational_dataset_5","Relational Dataset_5"],
    "P_form_name": ["Primary Form Name", "Form Name", "CRF Name", "primary_form_name", "form","Primary dataset Form"],
    "R1_form_name": ["relational_form_name_1","Relational Form Name_1","Relational Dataset Form"],
    "R2_form_name": ["relational_form_name_2","Relational Form Name_2"],
    "R3_form_name": ["relational_form_name_3","Relational Form Name_3"],
    "R4_form_name": ["relational_form_name_4","Relational Form Name_4"],
    "R5_form_name": ["relational_form_name_5","Relational Form Name_5"],
    "P_visit_name": ["Primary Visit Name", "Visit Name", "Visit", "primary_visit_name"],
    "R1_visit_name": ["relational_visit_name_1","Relational Visit Name_1"],
    "R2_visit_name": ["relational_visit_name_2","Relational Visit Name_2"],
    "R3_visit_name": ["relational_visit_name_3","Relational Visit Name_3"],
    "R4_visit_name": ["relational_visit_name_4","Relational Visit Name_4"],
    "R5_visit_name": ["relational_visit_name_5","Relational Visit Name_5"],
    # Variables
    "primary_dataset_columns": ["Primary Domain Variables", "Primary Dataset Columns", "Domain Variables", "Variables", "Primary Variables", "primary_dataset_columns","Primary dataset Variables"],
    "relational_dataset_variables": ["Relational Domain Variables", "Related Variables", "Relational Variables","Relational Dataset Variables"],
    "R1_Domain_Variables": ["relational_dataset_columns_1","Relational Dataset Columns_1"],
    "R2_Domain_Variables": ["relational_dataset_columns_2","Relational Dataset Columns_2"],
    "R3_Domain_Variables": ["relational_dataset_columns_3","Relational Dataset Columns_3"],
    "R4_Domain_Variables": ["relational_dataset_columns_4","Relational Dataset Columns_4"],
    "R5_Domain_Variables": ["relational_dataset_columns_5","Relational Dataset Columns_5"],
    # Relational dynamic columns (cleaned by MDDFileProcessor.clean_row_columns)
    "R1_Dynamic_Columns": ["relational_dynamic_columns_1", "Relational Dynamic Columns_1"],
    "R2_Dynamic_Columns": ["relational_dynamic_columns_2", "Relational Dynamic Columns_2"],
    "R3_Dynamic_Columns": ["relational_dynamic_columns_3", "Relational Dynamic Columns_3"],
    "R4_Dynamic_Columns": ["relational_dynamic_columns_4", "Relational Dynamic Columns_4"],
    "R5_Dynamic_Columns": ["relational_dynamic_columns_5", "Relational Dynamic Columns_5"],
    
    "dynamic_panel_variables": ["Dynamic Panel Variables", "Panel Variables", "Dynamic Variables","primary_dynamic_columns","Primary Dynamic Columns","Dynamic Column"],
    # Misc
    "query_target": ["Query Target", "Target", "Query Target Variable","query_target"],
    "origin_study": ["Origin Study", "Source Study", "Copy Source Study","study_id"],
    "pseudo_tech_code": ["Pseudo Tech Code", "Technical Code", "Tech Code","pseudo_tech_code"], #This Key is to generate NEW Check Logic
}

_INVALID_STRS = {"", "n/a", "na", "nan", "none"}

def _norm(val: Any) -> str:
    return str(val).strip().lower()

def get_field_value(row: Dict[str, Any], canonical_key: str) -> str:
    aliases = FIELD_ALIASES.get(canonical_key, [canonical_key])
    for alias in aliases:
        value = row.get(alias, "")
        if value and _norm(value) not in _INVALID_STRS:
            return str(value).strip()
    return ""

def has_any_value(row: Dict[str, Any], canonical_keys: List[str]) -> bool:
    return any(get_field_value(row, k) for k in canonical_keys)
