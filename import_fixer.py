#!/usr/bin/env python3
"""
Import Fixer Script for Quant Platform
Automatically detects and suggests fixes for import mismatches
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Set

class ImportFixer:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.available_classes = {}
        self.import_issues = []
        
    def scan_available_classes(self):
        """Scan all Python files to build a map of available classes"""
        print("🔍 Scanning available classes...")
        
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Find all class definitions
                class_matches = re.findall(r'^class\s+(\w+).*?:', content, re.MULTILINE)
                
                for class_name in class_matches:
                    module_path = str(py_file.relative_to(self.project_root)).replace('\\', '/').replace('.py', '').replace('/', '.')
                    
                    if class_name not in self.available_classes:
                        self.available_classes[class_name] = []
                    self.available_classes[class_name].append(module_path)
                    
            except Exception as e:
                print(f"❌ Error reading {py_file}: {e}")
        
        print(f"✅ Found {len(self.available_classes)} unique classes")
    
    def analyze_imports(self):
        """Analyze all import statements for issues"""
        print("\n🔍 Analyzing imports...")
        
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                for line_num, line in enumerate(lines, 1):
                    if line.strip().startswith('from ') and ' import ' in line:
                        self.check_import_line(py_file, line_num, line.strip())
                        
            except Exception as e:
                print(f"❌ Error analyzing {py_file}: {e}")
    
    def check_import_line(self, file_path: Path, line_num: int, import_line: str):
        """Check a specific import line for issues"""
        try:
            # Parse: from module import Class1, Class2, ...
            match = re.match(r'from\s+([\w\.]+)\s+import\s+(.+)', import_line)
            if not match:
                return
                
            module_path = match.group(1)
            imports = [imp.strip() for imp in match.group(2).split(',')]
            
            for imported_item in imports:
                # Clean up import (remove aliases, parentheses, etc.)
                clean_item = re.sub(r'\s+as\s+\w+|\(|\)', '', imported_item).strip()
                
                if clean_item in self.available_classes:
                    # Check if being imported from correct location
                    available_locations = self.available_classes[clean_item]
                    if module_path not in available_locations:
                        self.import_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'import_line': import_line,
                            'class': clean_item,
                            'wrong_module': module_path,
                            'correct_modules': available_locations,
                            'type': 'wrong_location'
                        })
                else:
                    # Check if class exists at all
                    if clean_item[0].isupper():  # Likely a class name
                        self.import_issues.append({
                            'file': str(file_path.relative_to(self.project_root)),
                            'line': line_num,
                            'import_line': import_line,
                            'class': clean_item,
                            'wrong_module': module_path,
                            'correct_modules': [],
                            'type': 'missing_class'
                        })
        except Exception as e:
            print(f"❌ Error checking import line: {e}")
    
    def generate_fixes(self):
        """Generate fix suggestions"""
        print(f"\n🔧 Found {len(self.import_issues)} import issues:")
        print("=" * 60)
        
        fixes_by_file = {}
        
        for issue in self.import_issues:
            file_name = issue['file']
            if file_name not in fixes_by_file:
                fixes_by_file[file_name] = []
            fixes_by_file[file_name].append(issue)
        
        for file_name, issues in fixes_by_file.items():
            print(f"\n📁 {file_name}:")
            print("-" * 40)
            
            for issue in issues:
                print(f"  Line {issue['line']}: {issue['import_line']}")
                
                if issue['type'] == 'wrong_location':
                    print(f"    ❌ {issue['class']} not found in {issue['wrong_module']}")
                    print(f"    ✅ Available in: {', '.join(issue['correct_modules'])}")
                    
                    # Suggest the best fix
                    best_module = self.suggest_best_module(issue['correct_modules'], issue['wrong_module'])
                    old_import = issue['import_line']
                    new_import = old_import.replace(issue['wrong_module'], best_module)
                    print(f"    🔧 Fix: {new_import}")
                    
                elif issue['type'] == 'missing_class':
                    print(f"    ❌ {issue['class']} does not exist anywhere")
                    print(f"    🔧 Options:")
                    print(f"       1. Remove from import")
                    print(f"       2. Create the class")
                    print(f"       3. Find similar class:")
                    
                    # Suggest similar classes
                    similar = self.find_similar_classes(issue['class'])
                    if similar:
                        print(f"          Similar: {', '.join(similar[:3])}")
                
                print()
    
    def suggest_best_module(self, available_modules: List[str], wrong_module: str) -> str:
        """Suggest the best module to import from"""
        # Prefer modules that are similar to the wrong one
        for module in available_modules:
            if any(part in module for part in wrong_module.split('.')):
                return module
        return available_modules[0]  # Fallback to first available
    
    def find_similar_classes(self, missing_class: str) -> List[str]:
        """Find classes with similar names"""
        similar = []
        missing_lower = missing_class.lower()
        
        for class_name in self.available_classes:
            if (missing_lower in class_name.lower() or 
                class_name.lower() in missing_lower or
                self.levenshtein_distance(missing_lower, class_name.lower()) <= 2):
                similar.append(class_name)
        
        return sorted(similar)[:5]
    
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

def main():
    print("🚀 Import Fixer for Quant Platform")
    print("=" * 50)
    
    fixer = ImportFixer()
    fixer.scan_available_classes()
    fixer.analyze_imports()
    fixer.generate_fixes()
    
    print("\n✅ Analysis complete!")
    print("Apply the suggested fixes one by one to resolve import issues.")

if __name__ == "__main__":
    main()