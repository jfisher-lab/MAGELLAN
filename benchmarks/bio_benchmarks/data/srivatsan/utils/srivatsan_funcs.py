import csv
import json
from pathlib import Path

import pandas as pd
import requests


def combine_header_lines(file_path: Path) -> list[str]:
    """
    Read and combine the two-line header from the file.
    
    The header is split across two lines where the first line has most columns
    but one column name 'model_component' is split as 'model_compon' and 'ent'.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        header_line1 = f.readline().strip()
        header_line2 = f.readline().strip()
    
    # Split both lines by whitespace
    cols1 = header_line1.split()
    cols2 = header_line2.split()
    
    # The last column in line 1 is incomplete ("model_compon")
    # and continues on line 2 ("ent"), followed by remaining columns
    if len(cols2) > 0:
        # Combine the split column name
        if len(cols1) > 0 and not cols1[-1].endswith('_'):
            # Merge last column of line 1 with first column of line 2
            cols1[-1] = cols1[-1] + cols2[0]
            # Add remaining columns from line 2
            cols1.extend(cols2[1:])
    
    return cols1

def extract_json_genes(json_path: Path) -> list[str]:
    """Extract gene names from BRCA pathway JSON model."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract variable names, excluding phenotypes and compound IDs
    genes = []
    for variable in data['Model']['Variables']:
        name = variable['Name']
        # Skip phenotype terms and compound identifiers
        if name not in ['Apoptosis', 'Proliferation', 'Survival'] and not name.startswith('C00'):
            genes.append(name)

    return genes

def load_srivatsan_drug_table(table_path: Path) -> pd.DataFrame:
    """Load and process supplementary table with compound information."""
    # Load tab-delimited file, skip the title line
    df = pd.read_csv(table_path, sep='\t', skiprows=1)

    # Create mapping from treatment to compound info
    # Keep only unique treatment mappings with compound names
    compound_map = df[['treatment', 'catalog_number', 'CAS.Number', 'name']].drop_duplicates()
    if not isinstance(compound_map, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(compound_map)}")
    if compound_map.empty:
        raise ValueError("Compound map DataFrame is empty")
    return compound_map

def classify_effect(row: pd.Series, significance: str = "q_value", sign: str = "normalized_effect") -> int:
    """
    Classify effect based on q-value and normalized effect.
    
    Args:
        row: DataFrame row with 'q_value' and 'normalized_effect' columns
        
    Returns:
        2 if significant and positive effect
        0 if significant and negative effect
        1 if not significant
    """
    if row[significance] < 0.05:
        return 2 if row[sign] > 0 else 0 if row[sign] < 0 else 1
    return 1

def get_pubchem_cid(cas_number: str) -> str | None:
    """Get PubChem CID from CAS number."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas_number}/cids/TXT"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200 and r.text.strip():
            return r.text.strip()
    except Exception as e:
        print(f"  Error getting CID for {cas_number}: {e}")
    return None

def get_chembl_id(cid: str) -> str | None:
    """Get ChEMBL ID from PubChem CID by searching synonyms."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            synonyms = data.get("InformationList", {}).get("Information", [{}])[0].get("Synonym", [])
            # Find ChEMBL IDs in synonyms
            chembl_ids = [s for s in synonyms if isinstance(s, str) and s.startswith("CHEMBL")]
            if chembl_ids:
                return chembl_ids[0]  # Return first ChEMBL ID found
    except Exception as e:
        print(f"  Error getting ChEMBL ID for CID {cid}: {e}")
    return None

def get_target_symbols(chembl_id: str) -> dict:
    """Query OpenTargets GraphQL API for target symbols and action types.

    Returns:
        dict with keys 'targets' (list of symbols) and 'uniqueActionTypes' (list of action types)
    """
    url = "https://api.platform.opentargets.org/api/v4/graphql"
    query = """
    query MechanismsOfActionSectionQuery($chemblId: String!) {
      drug(chemblId: $chemblId) {
        mechanismsOfAction {
          rows {
            targets {
              approvedSymbol
            }
          }
          uniqueActionTypes
        }
      }
    }
    """
    variables = {"chemblId": chembl_id}

    try:
        r = requests.post(url, json={"query": query, "variables": variables}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            drug_data = data.get("data", {}).get("drug", {})
            if not drug_data:
                return {"targets": [], "uniqueActionTypes": []}

            moa_data = drug_data.get("mechanismsOfAction", {})
            moa_rows = moa_data.get("rows", [])
            unique_action_types = moa_data.get("uniqueActionTypes", [])

            target_symbols = set()
            for row in moa_rows:
                for target in row.get("targets", []):
                    symbol = target.get("approvedSymbol")
                    if symbol:
                        target_symbols.add(symbol)

            return {
                "targets": sorted(list(target_symbols)),
                "uniqueActionTypes": unique_action_types if unique_action_types else []
            }
        else:
            print(f"  OpenTargets query failed for {chembl_id} (status {r.status_code})")
    except Exception as e:
        print(f"  Error querying OpenTargets for {chembl_id}: {e}")

    return {"targets": [], "uniqueActionTypes": []}

def load_unique_cas_numbers(csv_file: Path) -> list[str]:
    """Load unique CAS numbers from CSV file."""
    cas_numbers = set()
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cas = row.get("CAS.Number", "").strip()
            if cas:
                cas_numbers.add(cas)
    return sorted(list(cas_numbers))


def load_mcf7_basal_perturbations(literature_spec_file: Path) -> pd.DataFrame:
    """Load MCF7.basal perturbations from literature curated specification."""
    lit_spec = pd.read_csv(literature_spec_file)

    # Filter for MCF7.basal rows with perturbations only (not expectations)
    mcf7_basal = lit_spec.query("experiment_particular == 'MCF7.basal' and perturbation.notna()").copy()
    # Keep only gene and perturbation columns
    mcf7_basal = mcf7_basal.loc[:, ['gene', 'perturbation']].copy()

    print(f"  Loaded {len(mcf7_basal)} MCF7.basal perturbations:")
    for _, row in mcf7_basal.iterrows():
        print(f"    {row['gene']}: {row['perturbation']}")

    return mcf7_basal

def load_lookup_tables(
    cas_to_chembl_file: Path,
    chembl_to_targets_file: Path,
    brca_json_file: Path
) -> tuple[dict, dict, set[str]]:
    """Load CAS to ChEMBL, ChEMBL to targets, and BRCA genes."""
    with open(cas_to_chembl_file, 'r') as f:
        cas_to_chembl = json.load(f)

    with open(chembl_to_targets_file, 'r') as f:
        chembl_to_targets = json.load(f)

    brca_genes = set[str](extract_json_genes(brca_json_file))

    return dict(cas_to_chembl), dict(chembl_to_targets), brca_genes

def get_perturbation_level(action_types: list[str], chembl_id: str, brca_max_pert_level: int) -> int | None:
    """
    Determine perturbation level based on action types.

    Args:
        action_types: List of action types from ChEMBL
        chembl_id: ChEMBL ID of the compound
        brca_max_pert_level: Maximum perturbation level for activators

    Returns:
        0 for inhibitors/antagonists
        brca_max_pert_level for activators/agonists
        None if no valid action type or should be skipped
    """
    # Special case for CHEMBL1200675: Manually determined that this should have effect of zero in MCF7 cells
    if chembl_id == "CHEMBL1200675":
        return 0

    if not action_types:
        return None

    # Check for inhibitory actions
    if any(action.upper() in ["INHIBITOR", "ANTAGONIST"] for action in action_types):
        return 0

    # Check for activating actions
    if any(action.upper() in ["ACTIVATOR", "AGONIST"] for action in action_types):
        return brca_max_pert_level

    # For other action types (MODULATOR, STABILISER, etc.), skip
    return None

def create_target_perturbation_rows(
    cas_to_chembl: dict,
    chembl_to_targets: dict,
    brca_genes: set[str],
    df_filtered: pd.DataFrame,
    brca_max_pert_level: int
) -> pd.DataFrame:
    """Create rows for drug target perturbations."""

    perturbation_rows = []

    # Get unique CAS numbers and their drug names
    cas_drug_map = df_filtered[['CAS.Number', 'name']].drop_duplicates()

    drugs_with_targets = 0
    drugs_without_targets = 0
    total_target_rows = 0

    for _, row in cas_drug_map.iterrows():
        cas_number = row['CAS.Number']
        drug_name = row['name']

        # Look up ChEMBL ID
        chembl_id = cas_to_chembl.get(cas_number)
        if not chembl_id:
            print(f"  No ChEMBL ID for CAS {cas_number} ({drug_name})")
            drugs_without_targets += 1
            continue

        # Look up targets
        target_info = chembl_to_targets.get(chembl_id, {})
        targets = target_info.get('targets', [])
        action_types = target_info.get('uniqueActionTypes', [])

        # Filter targets to those in BRCA network
        valid_targets = [t for t in targets if t in brca_genes]

        if not valid_targets:
            print(f"  No BRCA targets for {drug_name} ({cas_number}, {chembl_id})")
            drugs_without_targets += 1
            continue

        # Determine perturbation level
        perturbation_level = get_perturbation_level(action_types, chembl_id, brca_max_pert_level)
        if perturbation_level is None:
            print(f"  No valid action type for {drug_name} ({chembl_id}): {action_types}")
            drugs_without_targets += 1
            continue

        drugs_with_targets += 1

        # Create a row for each target
        for target in valid_targets:
            perturbation_rows.append({
                'gene': target,
                'perturbation': perturbation_level,
                'expectation_bma': pd.NA,
                'drug_name': drug_name,
                'cas_number': cas_number,
                'chembl_id': chembl_id,
                'action_types': ', '.join(action_types) if action_types else ''
            })
            total_target_rows += 1

    print("\nDrug target summary:")
    print(f"  Drugs with valid targets: {drugs_with_targets}")
    print(f"  Drugs without valid targets: {drugs_without_targets}")
    print(f"  Total target perturbation rows: {total_target_rows}")

    return pd.DataFrame(perturbation_rows)