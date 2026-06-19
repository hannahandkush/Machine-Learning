import sys

def fix_notebook(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for i, line in enumerate(lines):
        # Remove empty lines that might have been left by sed
        if not line.strip():
            continue
        
        # Check if this line is a string but the next line starts a new key without a comma
        if i < len(lines) - 1:
            curr = line.strip()
            next_line = lines[i+1].strip()
            if curr.endswith('"') and next_line.startswith('"'):
                # Heuristic: if current line ends with quote and next starts with quote, add a comma
                line = line.replace('"\n', '",\n')
        
        fixed_lines.append(line)

    with open(file_path, 'w') as f:
        f.writelines(fixed_lines)

if __name__ == "__main__":
    fix_notebook(sys.argv[1])
