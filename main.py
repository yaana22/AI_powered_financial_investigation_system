import json

def convert_notebook_clean(input_ipynb, output_py):
    # Load the Jupyter notebook JSON structure
    with open(input_ipynb, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    clean_code_lines = []
    
    # Iterate through all cells in the notebook
    for cell in notebook.get('cells', []):
        # Only process code cells; ignore markdown completely
        if cell.get('cell_type') == 'code':
            lines = cell.get('source', [])
            
            for line in lines:
                # Omit Jupyter/IPython magic commands (%, !)
                if line.strip().startswith('%') or line.strip().startswith('!'):
                    continue
                clean_code_lines.append(line)
            
            # Add a trailing newline to separate code from different cells cleanly
            if clean_code_lines and not clean_code_lines[-1].endswith('\n'):
                clean_code_lines[-1] += '\n'
            clean_code_lines.append('\n')

    # Save the completely clean code to the target .py file
    with open(output_py, 'w', encoding='utf-8') as f:
        f.writelines(clean_code_lines)
        
    print(f"Successfully created clean script: {output_py}")

# Run the function (Replace with your file names)
convert_notebook_clean('./Ingestion/ingestion.ipynb', './Ingestion/new_Ingestion.py')
