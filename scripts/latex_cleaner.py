import sys, re

def clean_latex(s):
    # Remove redundant multipliers
    s = re.sub(r'1\.000 \\cdot ', '', s)
    s = re.sub(r' \\cdot 1\.000', '', s)
    s = re.sub(r'1\.0 \\cdot ', '', s)
    s = re.sub(r' \\cdot 1\.0', '', s)
    
    # Remove addition of zero
    s = re.sub(r'0\.000 \+ ', '', s)
    s = re.sub(r' \+ 0\.000', '', s)
    s = re.sub(r'0\.0 \+ ', '', s)
    s = re.sub(r' \+ 0\.0', '', s)
    
    # Clean signs
    s = s.replace('+ -', '-')
    s = s.replace('- -', '+')
    
    # Clean redundant parentheses (simple cases)
    s = re.sub(r'\(\((.*?)\)\)', r'(\1)', s)
    
    return s

if __name__ == "__main__":
    for line in sys.stdin:
        if line.startswith("$$"):
            print(clean_latex(line.strip()))
        else:
            print(line.strip())
