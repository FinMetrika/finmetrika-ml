# cli.py

import shutil
from pathlib import Path
import argparse


def create_notebook_from_template(template_path, 
                                  output_path):
    """
    Copy a template notebook to a new location.
    
    Parameters:
    - template_path: Path to the template notebook.
    - output_path: Path where the new notebook will be saved.
    """
    shutil.copy(template_path, output_path)
    print(f"New notebook created at {output_path}")



def main():
    parser = argparse.ArgumentParser(description="Create a new IPython notebook from a template.")
    parser.add_argument("output", help="Path where the new notebook will be saved.")
    args = parser.parse_args()

    # Adjust the path to the template as necessary
    # Ensure the template path is correctly referenced relative to this script's location
    template_path = Path(__file__).resolve().parent / "templates/template.ipynb"
    output_path = Path(args.output).resolve()

    create_notebook_from_template(template_path, output_path)



if __name__ == "__main__":
    main()
