#!/usr/bin/env python3
"""
Jupyter Notebook to LaTeX Chapter Converter
Converts Jupyter notebooks to LaTeX format compatible with book templates.
"""

import json
import re
from pathlib import Path


class JupyterToLatexConverter:
    """Converts Jupyter notebooks to LaTeX chapters."""

    def __init__(self, notebook_path: str, output_path: str):
        """Initialize and convert notebook to LaTeX."""
        self.notebook_path = Path(notebook_path)
        self.output_path = Path(output_path)
        self.convert()

    def convert(self):
        """Main conversion method."""
        # Load notebook
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Process content
        latex_content = []
        chapter_title = None

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                content = self.process_markdown(cell)
                if content:
                    # Extract first # as chapter title
                    if not chapter_title and content.startswith('\\section{'):
                        title_match = re.match(r'\\section\{(.+?)\}', content)
                        if title_match:
                            chapter_title = title_match.group(1)
                            content = content[len(title_match.group(0)):].strip()

                    if content:
                        latex_content.append(content)

            elif cell['cell_type'] == 'code':
                content = self.process_code(cell)
                if content:
                    latex_content.append(content)

        # Build final LaTeX
        output = f"% Generated from {self.notebook_path.name}\n\n"
        if chapter_title:
            output += f"\\chapter{{{chapter_title}}}\n\n"
        output += '\n\n'.join(latex_content)

        # Save to file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(output)

        print(f"✓ Converted: {self.notebook_path.name} → {self.output_path}")

    def process_markdown(self, cell):
        """Convert markdown cell to LaTeX."""
        text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if not text.strip():
            return ''

        # Headers (# is section since notebook is chapter)
        text = re.sub(r'^# (.+)$', r'\\section{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'\\subsection{\1}', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.+)$', r'\\subsubsection{\1}', text, flags=re.MULTILINE)

        # Text formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
        text = re.sub(r'`([^`]+)`', r'\\pyinline{\1}', text)

        # Math
        text = re.sub(r'\$\$(.+?)\$\$', r'\\[\1\\]', text, flags=re.DOTALL)
        text = re.sub(r'(?<!\$)\$(?!\$)([^\$]+)\$(?!\$)', r'$\1$', text)

        # Quotes
        text = re.sub(r'^> (.+)$', lambda m: self.format_quote(m.group(1)), text, flags=re.MULTILINE)

        # Lists
        text = self.process_lists(text)

        return text

    def format_quote(self, text):
        """Format block quote."""
        if ' - ' in text:
            parts = text.split(' - ', 1)
            return f'\\begin{{quote}}\n\\textit{{{parts[0].strip()}}}\n\\hfill --- {parts[1].strip()}\n\\end{{quote}}'
        return f'\\begin{{quote}}\n\\textit{{{text}}}\n\\end{{quote}}'

    def process_lists(self, text):
        """Convert markdown lists to LaTeX."""
        lines = text.split('\n')
        result = []
        in_list = False

        for line in lines:
            # Check for list items
            if re.match(r'^[-*] ', line):
                if not in_list:
                    result.append('\\begin{itemize}')
                    in_list = True
                result.append(re.sub(r'^[-*] ', r'\\item ', line))
            elif re.match(r'^\d+\. ', line):
                if not in_list:
                    result.append('\\begin{enumerate}')
                    in_list = True
                result.append(re.sub(r'^\d+\. ', r'\\item ', line))
            else:
                if in_list and line.strip() == '':
                    # Empty line might continue list
                    result.append(line)
                elif in_list:
                    # End list
                    env = 'enumerate' if '\\begin{enumerate}' in '\n'.join(result) else 'itemize'
                    result.append(f'\\end{{{env}}}')
                    in_list = False
                    result.append(line)
                else:
                    result.append(line)

        if in_list:
            env = 'enumerate' if '\\begin{enumerate}' in '\n'.join(result) else 'itemize'
            result.append(f'\\end{{{env}}}')

        return '\n'.join(result)

    def process_code(self, cell):
        """Convert code cell to LaTeX."""
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if not source.strip():
            return ''

        # Start with code block
        latex = "\\begin{pythoncode}\n"
        latex += source.strip()
        latex += "\n\\end{pythoncode}"

        # Add output if present
        outputs = self.extract_output(cell)
        if outputs:
            latex += "\n\\begin{mdframed}[backgroundcolor=gray!10,linecolor=gray!50]\n"
            latex += "\\textbf{Izlaz:}\n"
            latex += "\\begin{verbatim}\n"
            latex += outputs
            latex += "\n\\end{verbatim}\n"
            latex += "\\end{mdframed}"

        return latex

    def extract_output(self, cell):
        """Extract text output from code cell."""
        if 'outputs' not in cell:
            return ''

        output_lines = []
        for output in cell['outputs']:
            if output.get('output_type') == 'stream' and 'text' in output:
                text = output['text']
                output_lines.extend(text if isinstance(text, list) else [text])
            elif output.get('output_type') == 'execute_result':
                if 'data' in output and 'text/plain' in output['data']:
                    text = output['data']['text/plain']
                    output_lines.extend(text if isinstance(text, list) else [text])

        return ''.join(output_lines).strip()


if __name__ == '__main__':
    # Convert the notebook
    JupyterToLatexConverter('biljeznice/wittgenstein.ipynb', 'knjiga/wittgenstein.tex')