#!/usr/bin/env python3
"""
Jupyter Notebook to LaTeX Chapter Converter
Converts Jupyter notebooks to LaTeX format compatible with book templates.
Modified to support page-breaking code blocks, proper numbered header handling,
ASCII table conversion, and comprehensive Unicode symbol conversion to LaTeX.
"""

import json
import re
from pathlib import Path


class JupyterToLatexConverter:
    """Converts Jupyter notebooks to LaTeX chapters with breakable code blocks, tables, and symbol conversion."""

    def __init__(self, notebook_path: str, output_path: str):
        """Initialize and convert notebook to LaTeX."""
        self.notebook_path = Path(notebook_path)
        self.output_path = Path(output_path)

        # Comprehensive symbol mapping
        self.symbol_map = {
            # Logic symbols
            '⊥': r'$\bot$',
            '⊤': r'$\top$',
            '¬': r'$\neg$',
            '∧': r'$\land$',
            '∨': r'$\lor$',
            '→': r'$\rightarrow$',
            '←': r'$\leftarrow$',
            '↔': r'$\leftrightarrow$',
            '⇒': r'$\Rightarrow$',
            '⇐': r'$\Leftarrow$',
            '⇔': r'$\Leftrightarrow$',
            '∀': r'$\forall$',
            '∃': r'$\exists$',
            '∄': r'$\nexists$',
            '⊢': r'$\vdash$',  # syntactic entailment
            '⊨': r'$\models$',  # semantic entailment
            '⊣': r'$\dashv$',
            '⊩': r'$\Vdash$',
            '⊬': r'$\nvdash$',
            '⊭': r'$\nvDash$',
            '⊮': r'$\nVdash$',
            '⊯': r'$\nVDash$',

            # Set theory symbols
            '∈': r'$\in$',
            '∉': r'$\notin$',
            '⊂': r'$\subset$',
            '⊃': r'$\supset$',
            '⊆': r'$\subseteq$',
            '⊇': r'$\supseteq$',
            '⊈': r'$\nsubseteq$',
            '⊉': r'$\nsupseteq$',
            '∪': r'$\cup$',
            '∩': r'$\cap$',
            '∅': r'$\emptyset$',
            '⊕': r'$\oplus$',
            '⊗': r'$\otimes$',
            '⊖': r'$\ominus$',
            '⊙': r'$\odot$',

            # Mathematical operators
            '×': r'$\times$',
            '÷': r'$\div$',
            '±': r'$\pm$',
            '∓': r'$\mp$',
            '≠': r'$\neq$',
            '≡': r'$\equiv$',
            '≢': r'$\not\equiv$',
            '≈': r'$\approx$',
            '≉': r'$\not\approx$',
            '≤': r'$\leq$',
            '≥': r'$\geq$',
            '≪': r'$\ll$',
            '≫': r'$\gg$',
            '≺': r'$\prec$',
            '≻': r'$\succ$',
            '≼': r'$\preceq$',
            '≽': r'$\succeq$',
            '∝': r'$\propto$',
            '∞': r'$\infty$',
            '∂': r'$\partial$',
            '∇': r'$\nabla$',
            '∆': r'$\Delta$',
            '∑': r'$\sum$',
            '∏': r'$\prod$',
            '∫': r'$\int$',
            '∬': r'$\iint$',
            '∭': r'$\iiint$',
            '∮': r'$\oint$',

            # Arrows
            '↑': r'$\uparrow$',
            '↓': r'$\downarrow$',
            '↕': r'$\updownarrow$',
            '⇑': r'$\Uparrow$',
            '⇓': r'$\Downarrow$',
            '⇕': r'$\Updownarrow$',
            '↦': r'$\mapsto$',
            '↪': r'$\hookrightarrow$',

            # Greek letters (common ones)
            'α': r'$\alpha$',
            'β': r'$\beta$',
            'γ': r'$\gamma$',
            'δ': r'$\delta$',
            'ε': r'$\varepsilon$',
            'ζ': r'$\zeta$',
            'η': r'$\eta$',
            'θ': r'$\theta$',
            'ι': r'$\iota$',
            'κ': r'$\kappa$',
            'λ': r'$\lambda$',
            'μ': r'$\mu$',
            'ν': r'$\nu$',
            'ξ': r'$\xi$',
            'π': r'$\pi$',
            'ρ': r'$\rho$',
            'σ': r'$\sigma$',
            'τ': r'$\tau$',
            'υ': r'$\upsilon$',
            'φ': r'$\phi$',
            'χ': r'$\chi$',
            'ψ': r'$\psi$',
            'ω': r'$\omega$',
            'Γ': r'$\Gamma$',
            'Δ': r'$\Delta$',
            'Θ': r'$\Theta$',
            'Λ': r'$\Lambda$',
            'Ξ': r'$\Xi$',
            'Π': r'$\Pi$',
            'Σ': r'$\Sigma$',
            'Φ': r'$\Phi$',
            'Ψ': r'$\Psi$',
            'Ω': r'$\Omega$',

            # Other useful symbols
            '•': r'$\bullet$',
            '∘': r'$\circ$',
            '°': r'$^\circ$',
            '†': r'$\dagger$',
            '‡': r'$\ddagger$',
            '★': r'$\star$',
            '♠': r'$\spadesuit$',
            '♣': r'$\clubsuit$',
            '♥': r'$\heartsuit$',
            '♦': r'$\diamondsuit$',
            '□': r'$\square$',
            '■': r'$\blacksquare$',
            '◊': r'$\lozenge$',
            '○': r'$\bigcirc$',
            '●': r'$\bullet$',
            '⟨': r'$\langle$',
            '⟩': r'$\rangle$',
            '⌊': r'$\lfloor$',
            '⌋': r'$\rfloor$',
            '⌈': r'$\lceil$',
            '⌉': r'$\rceil$',
            '〈': r'$\langle$',
            '〉': r'$\rangle$',
        }

        self.convert()

    def convert_symbols(self, text: str, in_math_mode: bool = False) -> str:
        """Convert Unicode symbols to LaTeX equivalents.

        Args:
            text: The text to convert
            in_math_mode: If True, don't wrap symbols in $ signs (already in math mode)
        """
        if not text:
            return text

        # If we're in math mode, create a version of the map without dollar signs
        if in_math_mode:
            for symbol, latex in self.symbol_map.items():
                # Remove the dollar signs for math mode
                latex_no_dollars = latex.replace('$', '')
                text = text.replace(symbol, latex_no_dollars)
        else:
            # Regular replacement with dollar signs
            for symbol, latex in self.symbol_map.items():
                text = text.replace(symbol, latex)

        return text

    def clean_header(self, text: str) -> str:
        """Clean header text by removing leading numbers like '1.' but preserving the rest."""
        # Remove patterns like "1. " or "1.2. " from the beginning
        cleaned = re.sub(r'^(\d+\.)+\s*', '', text.strip())
        return cleaned.strip()

    def detect_and_convert_table(self, text: str) -> str:
        """Detect ASCII tables and convert them to LaTeX format."""
        lines = text.split('\n')
        result = []
        i = 0

        while i < len(lines):
            # Check if this looks like the start of a table
            if self.is_table_start(lines, i):
                # Extract and convert the table
                table_latex, end_index = self.extract_and_convert_table(lines, i)
                if table_latex:
                    result.append(table_latex)
                    i = end_index + 1
                    continue

            result.append(lines[i])
            i += 1

        return '\n'.join(result)

    def is_table_start(self, lines, start_index):
        """Check if the current position is the start of a table."""
        if start_index >= len(lines):
            return False

        # Look for patterns that indicate a table
        # Could be a line with pipes, or a separator line
        for i in range(start_index, min(start_index + 5, len(lines))):
            line = lines[i].strip()
            # Check for table markers
            if '|' in line or re.match(r'^[=\-]+$', line):
                # Look ahead to confirm it's a table
                for j in range(i, min(i + 3, len(lines))):
                    if '|' in lines[j]:
                        return True
        return False

    def extract_and_convert_table(self, lines, start_index):
        """Extract a table from lines and convert to LaTeX."""
        table_lines = []
        end_index = start_index

        # Find the table boundaries
        for i in range(start_index, len(lines)):
            line = lines[i].strip()

            # Skip empty lines at the beginning
            if not line and not table_lines:
                continue

            # Check for table content or separators
            if '|' in line or re.match(r'^[=\-]+$', line) or (table_lines and not line):
                if not line and table_lines:
                    # Empty line after table content might indicate end
                    # Check if there's more table content ahead
                    has_more = False
                    for j in range(i + 1, min(i + 3, len(lines))):
                        if '|' in lines[j] or re.match(r'^[=\-]+$', lines[j]):
                            has_more = True
                            break
                    if not has_more:
                        end_index = i - 1
                        break
                table_lines.append(line)
                end_index = i
            else:
                # If we have table lines and hit non-table content, stop
                if table_lines:
                    end_index = i - 1
                    break

        if not table_lines:
            return None, start_index

        # Parse the table
        rows = []
        headers = None

        for line in table_lines:
            line = line.strip()

            # Skip separator lines
            if re.match(r'^[=]+$', line) or re.match(r'^[\-]+$', line):
                continue

            # Parse data rows
            if '|' in line:
                # Split by pipe and clean up
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at the beginning and end
                while cells and not cells[0]:
                    cells.pop(0)
                while cells and not cells[-1]:
                    cells.pop()

                if cells:
                    # First row with cells is the header
                    if headers is None:
                        headers = cells
                    else:
                        rows.append(cells)

        if not headers:
            return None, start_index

        # Convert to LaTeX
        latex = self.format_latex_table(headers, rows)
        return latex, end_index

    def format_latex_table(self, headers, rows):
        """Format a table as LaTeX."""
        num_cols = len(headers)

        # Start table
        latex = "\\begin{table}[h!]\n"
        latex += "\\centering\n"
        latex += "\\begin{tabular}{" + "|c" * num_cols + "|}\n"
        latex += "\\hline\n"

        # Add headers with symbol conversion
        header_cells = [self.convert_symbols(h) for h in headers]
        latex += " & ".join(header_cells) + " \\\\\n"
        latex += "\\hline\n"

        # Add rows with symbol conversion
        for row in rows:
            # Ensure row has the right number of cells
            while len(row) < num_cols:
                row.append('')
            row_cells = [self.convert_symbols(cell) for cell in row[:num_cols]]
            latex += " & ".join(row_cells) + " \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

        return latex

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

        print(f"✔ Converted: {self.notebook_path.name} → {self.output_path}")

    def process_markdown(self, cell):
        """Convert markdown cell to LaTeX."""
        text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if not text.strip():
            return ''

        # First, detect and convert tables (they handle symbols internally)
        text = self.detect_and_convert_table(text)

        # Convert symbols in the entire text (but preserve existing LaTeX math)
        # We need to be careful not to convert symbols inside existing $ ... $ or \[ ... \]
        text = self.convert_symbols_preserve_math(text)

        # Process headers with number cleaning
        # Using lambda functions to clean the captured header text
        text = re.sub(
            r'^# (.+)$',
            lambda m: f'\\section{{{self.clean_header(m.group(1))}}}',
            text,
            flags=re.MULTILINE
        )
        text = re.sub(
            r'^## (.+)$',
            lambda m: f'\\section{{{self.clean_header(m.group(1))}}}',
            text,
            flags=re.MULTILINE
        )
        text = re.sub(
            r'^### (.+)$',
            lambda m: f'\\subsection{{{self.clean_header(m.group(1))}}}',
            text,
            flags=re.MULTILINE
        )
        text = re.sub(
            r'^#### (.+)$',
            lambda m: f'\\subsubsection{{{self.clean_header(m.group(1))}}}',
            text,
            flags=re.MULTILINE
        )

        # Text formatting
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
        text = re.sub(r'`([^`]+)`', r'\\pyinline{\1}', text)

        # Math (handle display math first, then inline)
        text = re.sub(r'\$\$(.+?)\$\$', r'\\[\1\\]', text, flags=re.DOTALL)
        text = re.sub(r'(?<!\$)\$(?!\$)([^\$]+)\$(?!\$)', r'$\1$', text)

        # Quotes
        text = re.sub(r'^> (.+)$', lambda m: self.format_quote(m.group(1)), text, flags=re.MULTILINE)

        # Lists
        text = self.process_lists(text)

        # Links (convert markdown links to LaTeX)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\\href{\2}{\1}', text)

        return text

    def convert_symbols_preserve_math(self, text: str) -> str:
        """Convert symbols while preserving existing math environments."""
        # Split text by math delimiters
        parts = []
        current_pos = 0

        # Find all math regions (both $...$ and \[...\])
        math_regions = []

        # Find inline math $...$
        for match in re.finditer(r'\$[^\$]+\$', text):
            math_regions.append((match.start(), match.end(), 'inline'))

        # Find display math \[...\]
        for match in re.finditer(r'\\\[.*?\\\]', text, re.DOTALL):
            math_regions.append((match.start(), match.end(), 'display'))

        # Find display math $$...$$
        for match in re.finditer(r'\$\$.*?\$\$', text, re.DOTALL):
            math_regions.append((match.start(), match.end(), 'display'))

        # Sort regions by start position
        math_regions.sort(key=lambda x: x[0])

        # Process text, converting symbols outside math regions
        result = []
        last_end = 0

        for start, end, math_type in math_regions:
            # Convert symbols in text before this math region
            if start > last_end:
                before_math = text[last_end:start]
                result.append(self.convert_symbols(before_math, in_math_mode=False))

            # Keep math region as-is (it should already have proper LaTeX)
            result.append(text[start:end])
            last_end = end

        # Convert symbols in remaining text after last math region
        if last_end < len(text):
            remaining = text[last_end:]
            result.append(self.convert_symbols(remaining, in_math_mode=False))

        return ''.join(result)

    def format_quote(self, text):
        """Format block quote."""
        # Convert symbols in quote text
        text = self.convert_symbols(text)

        if ' - ' in text:
            parts = text.split(' - ', 1)
            return f'\\begin{{quote}}\n\\textit{{{parts[0].strip()}}}\n\\hfill --- {parts[1].strip()}\n\\end{{quote}}'
        return f'\\begin{{quote}}\n\\textit{{{text}}}\n\\end{{quote}}'

    def process_lists(self, text):
        """Convert markdown lists to LaTeX."""
        lines = text.split('\n')
        result = []
        in_list = False
        list_type = None
        indent_stack = []  # Track nested lists

        for line in lines:
            # Check indentation level
            indent = len(line) - len(line.lstrip())

            # Check for unordered list items
            unordered_match = re.match(r'^(\s*)[-*+] (.+)$', line)
            ordered_match = re.match(r'^(\s*)\d+\. (.+)$', line)

            if unordered_match:
                current_indent = len(unordered_match.group(1))
                item_content = unordered_match.group(2)
                # Convert symbols in list item
                item_content = self.convert_symbols(item_content)

                if not in_list:
                    result.append('\\begin{itemize}')
                    in_list = True
                    list_type = 'itemize'
                    indent_stack = [current_indent]

                # Handle nested lists
                if current_indent > indent_stack[-1]:
                    result.append('\\begin{itemize}')
                    indent_stack.append(current_indent)
                elif current_indent < indent_stack[-1]:
                    while indent_stack and current_indent < indent_stack[-1]:
                        result.append('\\end{itemize}')
                        indent_stack.pop()

                result.append(f'\\item {item_content}')

            elif ordered_match:
                current_indent = len(ordered_match.group(1))
                item_content = ordered_match.group(2)
                # Convert symbols in list item
                item_content = self.convert_symbols(item_content)

                if not in_list:
                    result.append('\\begin{enumerate}')
                    in_list = True
                    list_type = 'enumerate'
                    indent_stack = [current_indent]

                # Handle nested lists
                if current_indent > indent_stack[-1]:
                    result.append('\\begin{enumerate}')
                    indent_stack.append(current_indent)
                elif current_indent < indent_stack[-1]:
                    while indent_stack and current_indent < indent_stack[-1]:
                        result.append('\\end{enumerate}')
                        indent_stack.pop()

                result.append(f'\\item {item_content}')

            else:
                # Check if it's a continuation of a list item (indented text)
                if in_list and line.strip() and indent > 0:
                    result.append(line.strip())
                elif in_list and line.strip() == '':
                    # Empty line might continue list
                    result.append('')
                elif in_list:
                    # End all nested lists
                    while indent_stack:
                        if list_type == 'enumerate':
                            result.append('\\end{enumerate}')
                        else:
                            result.append('\\end{itemize}')
                        indent_stack.pop()
                    in_list = False
                    list_type = None
                    result.append(line)
                else:
                    result.append(line)

        # Close any remaining lists
        if in_list:
            while indent_stack:
                if list_type == 'enumerate':
                    result.append('\\end{enumerate}')
                else:
                    result.append('\\end{itemize}')
                indent_stack.pop()

        return '\n'.join(result)

    def process_code(self, cell):
        """Convert code cell to LaTeX with breakable blocks."""
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if not source.strip():
            return ''

        # Convert any symbols in code comments (but be careful with string literals)
        source = self.convert_symbols_in_code(source)

        # Start with code block (pythoncode environment is already breakable)
        latex = "\\begin{pythoncode}\n"
        latex += source.strip()
        latex += "\n\\end{pythoncode}"

        # Add output if present using tcolorbox (breakable)
        outputs = self.extract_output(cell)
        if outputs:
            # Check if output contains a table
            if self.looks_like_table_output(outputs):
                # Convert the table in the output
                converted_output = self.detect_and_convert_table(outputs)
                # If a table was found and converted, use it directly
                if '\\begin{table}' in converted_output:
                    latex += "\n" + converted_output
                else:
                    # Otherwise, use regular output formatting with symbol conversion
                    outputs = self.convert_symbols(outputs)
                    latex += "\n\\begin{codeoutput}\n"
                    latex += "\\begin{verbatim}\n"
                    latex += outputs
                    latex += "\n\\end{verbatim}\n"
                    latex += "\\end{codeoutput}"
            else:
                # Convert symbols in output
                outputs = self.convert_symbols(outputs)
                # Use tcolorbox-based codeoutput environment instead of mdframed
                latex += "\n\\begin{codeoutput}\n"
                latex += "\\begin{verbatim}\n"
                latex += outputs
                latex += "\n\\end{verbatim}\n"
                latex += "\\end{codeoutput}"

        return latex

    def convert_symbols_in_code(self, code: str) -> str:
        """Convert symbols in code, being careful with string literals and comments."""
        lines = code.split('\n')
        result = []

        for line in lines:
            # Only convert symbols in comments (after #)
            if '#' in line:
                code_part, comment_part = line.split('#', 1)
                comment_part = self.convert_symbols(comment_part)
                result.append(code_part + '#' + comment_part)
            else:
                # For now, don't convert symbols in actual code to avoid breaking string literals
                result.append(line)

        return '\n'.join(result)

    def looks_like_table_output(self, text):
        """Check if the output looks like a table."""
        lines = text.split('\n')
        pipe_count = 0
        separator_count = 0

        for line in lines:
            if '|' in line:
                pipe_count += 1
            if re.match(r'^[=\-]+$', line.strip()):
                separator_count += 1

        # If we have multiple lines with pipes and some separators, it's likely a table
        return pipe_count >= 2 and separator_count >= 1

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
            elif output.get('output_type') == 'error':
                # Include error outputs as well
                if 'traceback' in output:
                    # Traceback is usually a list of strings with ANSI codes
                    for line in output['traceback']:
                        # Remove ANSI escape codes
                        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                        output_lines.append(clean_line)

        return ''.join(output_lines).strip()

if __name__ == '__main__':
    # Convert the notebook
    JupyterToLatexConverter('biljeznice/wittgenstein.ipynb', 'knjiga/wittgenstein.tex')
    JupyterToLatexConverter('biljeznice/gentzen.ipynb', 'knjiga/gentzen.tex')
    JupyterToLatexConverter('biljeznice/tarski.ipynb', 'knjiga/tarski.tex')