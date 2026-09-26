#!/usr/bin/env python3
"""
Submission Packaging Helper for ML Challenge 2026.
Validates submission TSV files and packages the required zip structure:
  <team_name>_submission.zip
  ├── output/
  │   ├── matching_results.tsv
  │   └── candidate_pairs.tsv
  ├── code/
  │   └── business_entity_resolution/
  │       ├── src/
  │       ├── pipeline.py
  │       ├── README.md
  │       └── requirements.txt
  └── Documentation_template.md
"""

import argparse
import os
import subprocess
import sys
import zipfile


def package_submission(team_name: str, test_dir: str, skip_val: bool = False):
    output_dir = "output"
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    code_dir = os.path.join("code", "business_entity_resolution")
    doc_file = "Documentation_template.md"

    # 1. Check file existence
    if not os.path.exists(matching_file):
        print(f"Error: Required file not found: {matching_file}")
        sys.exit(1)
    if not os.path.exists(candidate_file):
        print(f"Error: Required file not found: {candidate_file}")
        sys.exit(1)
    if not os.path.exists(code_dir):
        print(f"Error: Code directory not found: {code_dir}")
        sys.exit(1)
    if not os.path.exists(doc_file):
        print(f"Error: Methodology document not found: {doc_file}")
        sys.exit(1)

    # 2. Run validator
    if not skip_val:
        validator_path = os.path.join("student_resource", "utils", "validate_submission.py")
        if not os.path.exists(validator_path):
            validator_path = os.path.join("utils", "validate_submission.py")
        
        if os.path.exists(validator_path):
            print("Running submission validator before packaging...")
            cmd = [
                sys.executable,
                validator_path,
                "--matching", matching_file,
                "--candidate", candidate_file,
                "--test-dir", test_dir,
            ]
            ret = subprocess.call(cmd)
            if ret != 0:
                print(f"\nPackaging aborted: validator returned exit code {ret}.")
                sys.exit(ret)
            print("Validation successful!\n")

    # 3. Create Zip file
    clean_team_name = team_name.strip().replace(" ", "_")
    zip_filename = f"{clean_team_name}_submission.zip"

    print(f"Packaging submission archive: {zip_filename} ...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add output files
        zf.write(matching_file, arcname=f"output/{os.path.basename(matching_file)}")
        zf.write(candidate_file, arcname=f"output/{os.path.basename(candidate_file)}")

        # Add documentation
        zf.write(doc_file, arcname="Documentation_template.md")

        # Add code directory recursively
        for root, dirs, files in os.walk(code_dir):
            for file in files:
                if file.endswith((".pyc", ".DS_Store", ".joblib")) or "__pycache__" in root:
                    continue
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, "code")
                arcname = os.path.join("code", rel_path)
                zf.write(file_path, arcname=arcname)

    print(f"SUCCESS: Created submission package '{zip_filename}'!")
    print("Archive contents:")
    with zipfile.ZipFile(zip_filename, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename} ({info.file_size:,} bytes)")


def main():
    parser = argparse.ArgumentParser(description="Package ML Challenge submission zip")
    parser.add_argument("--team-name", default="Code_Alchemists", help="Your team name")
    parser.add_argument(
        "--test-dir",
        default="student_resource/dataset/test",
        help="Path to test set directory for validation",
    )
    parser.add_argument("--skip-validation", action="store_true", help="Skip running validator")
    args = parser.parse_args()

    package_submission(
        team_name=args.team_name,
        test_dir=args.test_dir,
        skip_val=args.skip_validation,
    )


if __name__ == "__main__":
    main()
