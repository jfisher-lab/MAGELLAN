import json
import time
from pathlib import Path

import fire

from benchmarks.bio_benchmarks.data.srivatsan.utils.srivatsan_funcs import (
    get_chembl_id,
    get_pubchem_cid,
    get_target_symbols,
    load_unique_cas_numbers,
)


def main(
    csv_file: str = str(
        Path(__file__).parent.parent / "mcf7_brca_filtered_processed.csv"
    ),
    cas_to_chembl_file: str = str(Path(__file__).parent.parent / "cas_to_chembl.json"),
    chembl_to_targets_file: str = str(
        Path(__file__).parent.parent / "chembl_to_targets.json"
    ),
    sleep_time: float = 0.5,
) -> None:
    """
    Annotate CAS numbers with ChEMBL IDs and target information.

    Args:
        csv_file: Path to input CSV file with CAS numbers
        cas_to_chembl_file: Path to output CAS to ChEMBL mapping JSON file
        chembl_to_targets_file: Path to output ChEMBL to targets mapping JSON file
        sleep_time: Time to sleep between API requests (seconds)
    """
    csv_path = Path(csv_file)
    cas_to_chembl_path = Path(cas_to_chembl_file)
    chembl_to_targets_path = Path(chembl_to_targets_file)

    print(f"Loading CAS numbers from {csv_path}...")
    cas_numbers = load_unique_cas_numbers(csv_path)
    print(f"Found {len(cas_numbers)} unique CAS numbers")

    # Step 1: Build CAS to ChEMBL ID mapping
    print("\n" + "=" * 60)
    print("STEP 1: Building CAS → ChEMBL ID lookup table")
    print("=" * 60)

    cas_to_chembl = {}
    chembl_ids = []

    for i, cas in enumerate(cas_numbers, 1):
        print(f"\n[{i}/{len(cas_numbers)}] Processing CAS: {cas}")

        cid = get_pubchem_cid(cas)
        if not cid:
            print("  ❌ No PubChem CID found")
            continue

        print(f"  ✓ PubChem CID: {cid}")

        chembl_id = get_chembl_id(cid)
        if not chembl_id:
            print("  ❌ No ChEMBL ID found")
            continue

        print(f"  ✓ ChEMBL ID: {chembl_id}")
        cas_to_chembl[cas] = chembl_id
        chembl_ids.append(chembl_id)

        # Be polite to the APIs
        time.sleep(sleep_time)

    # Save CAS to ChEMBL mapping
    with open(cas_to_chembl_path, "w", encoding="utf-8") as f:
        json.dump(cas_to_chembl, f, indent=2)
    print(f"\n✓ Saved CAS → ChEMBL mapping to {cas_to_chembl_path}")
    print(f"  Mapped {len(cas_to_chembl)} out of {len(cas_numbers)} CAS numbers")

    # Step 2: Build ChEMBL ID to target symbols mapping
    print("\n" + "=" * 60)
    print("STEP 2: Building ChEMBL ID → Target Symbols lookup table")
    print("=" * 60)

    # Get unique ChEMBL IDs
    unique_chembl_ids = sorted(list(set(chembl_ids)))
    print(f"Found {len(unique_chembl_ids)} unique ChEMBL IDs")

    chembl_to_targets = {}

    for i, chembl_id in enumerate(unique_chembl_ids, 1):
        print(f"\n[{i}/{len(unique_chembl_ids)}] Processing ChEMBL ID: {chembl_id}")

        result = get_target_symbols(chembl_id)
        targets = result["targets"]
        action_types = result["uniqueActionTypes"]

        if targets:
            action_str = (
                f" (actions: {', '.join(action_types)})" if action_types else ""
            )
            print(
                f"  ✓ Found {len(targets)} target(s): {', '.join(targets)}{action_str}"
            )
            chembl_to_targets[chembl_id] = result
        else:
            print("  ⚠ No targets found")
            chembl_to_targets[chembl_id] = result

        # Be polite to the APIs
        time.sleep(sleep_time)

    # Save ChEMBL to targets mapping
    with open(chembl_to_targets_path, "w", encoding="utf-8") as f:
        json.dump(chembl_to_targets, f, indent=2)
    print(f"\n✓ Saved ChEMBL → Targets mapping to {chembl_to_targets_path}")
    print(
        f"  Found targets for {sum(1 for v in chembl_to_targets.values() if v['targets'])} out of {len(chembl_to_targets)} ChEMBL IDs"
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total CAS numbers: {len(cas_numbers)}")
    print(f"CAS → ChEMBL mappings: {len(cas_to_chembl)}")
    print(f"Unique ChEMBL IDs: {len(unique_chembl_ids)}")
    print(
        f"ChEMBL IDs with targets: {sum(1 for v in chembl_to_targets.values() if v['targets'])}"
    )
    print(
        f"ChEMBL IDs with action types: {sum(1 for v in chembl_to_targets.values() if v['uniqueActionTypes'])}"
    )
    print("\nOutput files:")
    print(f"  - {cas_to_chembl_path}")
    print(f"  - {chembl_to_targets_path}")


if __name__ == "__main__":
    fire.Fire(main)
