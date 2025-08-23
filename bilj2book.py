#!/usr/bin/env python3
"""
Jupyter Notebook to LaTeX Chapter Converter
Converts Jupyter notebooks to LaTeX format compatible with book templates.
Enhanced to detect and properly render LaTeX content in code outputs.
"""

import json
import re
from pathlib import Path


class JupyterToLatexConverter:
    """Converts Jupyter notebooks to LaTeX chapters with intelligent LaTeX detection."""

    def __init__(self, notebook_path: str, output_path: str):
        """Initialize and convert notebook to LaTeX."""
        self.notebook_path = Path(notebook_path)
        self.output_path = Path(output_path)

        # Comprehensive symbol mapping (without dollar signs)
        self.symbol_map = {
            # Logic symbols
            '⊥': r'\bot',
            '⊤': r'\top',
            '¬': r'\neg',
            '∧': r'\land',
            '∨': r'\lor',
            '→': r'\rightarrow',
            '←': r'\leftarrow',
            '↔': r'\leftrightarrow',
            '⇒': r'\Rightarrow',
            '⇐': r'\Leftarrow',
            '⇔': r'\Leftrightarrow',
            '∀': r'\forall',
            '∃': r'\exists',
            '∄': r'\nexists',
            '⊢': r'\vdash',
            '⊨': r'\models',
            '⊣': r'\dashv',
            '⊩': r'\Vdash',
            '⊬': r'\nvdash',
            '⊭': r'\nvDash',
            '⊮': r'\nVdash',
            '⊯': r'\nVDash',

            # Set theory symbols
            '∈': r'\in',
            '∉': r'\notin',
            '⊂': r'\subset',
            '⊃': r'\supset',
            '⊆': r'\subseteq',
            '⊇': r'\supseteq',
            '⊈': r'\nsubseteq',
            '⊉': r'\nsupseteq',
            '∪': r'\cup',
            '∩': r'\cap',
            '∅': r'\emptyset',
            '⊕': r'\oplus',
            '⊗': r'\otimes',
            '⊖': r'\ominus',
            '⊙': r'\odot',

            # Mathematical operators
            '×': r'\times',
            '÷': r'\div',
            '±': r'\pm',
            '∓': r'\mp',
            '≠': r'\neq',
            '≡': r'\equiv',
            '≢': r'\not\equiv',
            '≈': r'\approx',
            '≉': r'\not\approx',
            '≤': r'\leq',
            '≥': r'\geq',
            '≪': r'\ll',
            '≫': r'\gg',
            '≺': r'\prec',
            '≻': r'\succ',
            '≼': r'\preceq',
            '≽': r'\succeq',
            '∝': r'\propto',
            '∞': r'\infty',
            '∂': r'\partial',
            '∇': r'\nabla',
            '∆': r'\Delta',
            '∑': r'\sum',
            '∏': r'\prod',
            '∫': r'\int',
            '∬': r'\iint',
            '∭': r'\iiint',
            '∮': r'\oint',

            # Arrows
            '↑': r'\uparrow',
            '↓': r'\downarrow',
            '↕': r'\updownarrow',
            '⇑': r'\Uparrow',
            '⇓': r'\Downarrow',
            '⇕': r'\Updownarrow',
            '↦': r'\mapsto',
            '↪': r'\hookrightarrow',

            # Greek letters
            'α': r'\alpha',
            'β': r'\beta',
            'γ': r'\gamma',
            'δ': r'\delta',
            'ε': r'\varepsilon',
            'ζ': r'\zeta',
            'η': r'\eta',
            'θ': r'\theta',
            'ι': r'\iota',
            'κ': r'\kappa',
            'λ': r'\lambda',
            'μ': r'\mu',
            'ν': r'\nu',
            'ξ': r'\xi',
            'π': r'\pi',
            'ρ': r'\rho',
            'σ': r'\sigma',
            'τ': r'\tau',
            'υ': r'\upsilon',
            'φ': r'\phi',
            'χ': r'\chi',
            'ψ': r'\psi',
            'ω': r'\omega',
            'Γ': r'\Gamma',
            'Δ': r'\Delta',
            'Θ': r'\Theta',
            'Λ': r'\Lambda',
            'Ξ': r'\Xi',
            'Π': r'\Pi',
            'Σ': r'\Sigma',
            'Φ': r'\Phi',
            'Ψ': r'\Psi',
            'Ω': r'\Omega',

            # Other useful symbols
            '•': r'\bullet',
            '∘': r'\circ',
            '°': r'^\circ',
            '†': r'\dagger',
            '‡': r'\ddagger',
            '★': r'\star',
            '□': r'\square',
            '■': r'\blacksquare',
            '◊': r'\lozenge',
            '○': r'\bigcirc',
            '●': r'\bullet',
            '⟨': r'\langle',
            '⟩': r'\rangle',
            '⌊': r'\lfloor',
            '⌋': r'\rfloor',
            '⌈': r'\lceil',
            '⌉': r'\rceil',
            '〈': r'\langle',
            '〉': r'\rangle',
        }

        self.convert()

    def is_latex_content(self, text: str) -> bool:
        """Check if text contains LaTeX commands that should be rendered."""
        latex_indicators = [
            r'\\begin\{prooftree\}',
            r'\\begin\{equation',
            r'\\begin\{align',
            r'\\begin\{gather',
            r'\\begin\{array',
            r'\\begin\{matrix',
            r'\\begin\{bmatrix',
            r'\\begin\{pmatrix',
            r'\\AxiomC',
            r'\\UnaryInfC',
            r'\\BinaryInfC',
            r'\\TrinaryInfC',
            r'\\RightLabel',
            r'\\LeftLabel',
            r'\\frac\{',
            r'\\displaystyle',
            r'\\[A-Z][a-z]+C\{',  # Catches inference commands
        ]

        for pattern in latex_indicators:
            if re.search(pattern, text):
                return True

        # Also check if it looks like a structured LaTeX example
        if ('\\wedge' in text or '\\vee' in text or '\\neg' in text or
                '\\rightarrow' in text or '\\vdash' in text or '\\models' in text):
            if ('\\begin{' in text or '\\end{' in text):
                return True

        return False

    def process_latex_output(self, text: str) -> str:
        """Process output that contains LaTeX code."""
        lines = text.split('\n')
        result = []
        in_latex_block = False
        latex_buffer = []
        header_lines = []

        for line in lines:
            # Check for header/separator lines
            if re.match(r'^[=\-]{3,}$', line.strip()):
                if latex_buffer:
                    # Process accumulated LaTeX
                    latex_content = '\n'.join(latex_buffer)
                    result.append(self.convert_latex_content(latex_content))
                    latex_buffer = []
                    in_latex_block = False
                continue

            # Check for section headers (like "KONJUNKCIJA:")
            if re.match(r'^[A-Z][A-Z\s]+:$', line.strip()) and not in_latex_block:
                if latex_buffer:
                    result.append(self.convert_latex_content('\n'.join(latex_buffer)))
                    latex_buffer = []
                result.append(f"\n\\textbf{{{line.strip()}}}\n")
                continue

            # Check for subsection headers (like "Uvođenje (∧I):")
            if re.match(r'^[A-Z][a-zščćžđ]+.*\([^)]+\):$', line.strip()):
                if latex_buffer:
                    result.append(self.convert_latex_content('\n'.join(latex_buffer)))
                    latex_buffer = []
                # Convert symbols in the header
                converted_line = self.convert_symbols_in_text(line.strip())
                result.append(f"\n\\textit{{{converted_line}}}\n")
                continue

            # Detect LaTeX environment beginnings
            if re.search(r'\\begin\{', line):
                in_latex_block = True
                latex_buffer.append(line)
            elif re.search(r'\\end\{', line):
                latex_buffer.append(line)
                # Check if this ends a major environment
                if re.search(r'\\end\{(prooftree|equation|align|gather|array|matrix)', line):
                    # Process the complete LaTeX block
                    latex_content = '\n'.join(latex_buffer)
                    result.append(self.convert_latex_content(latex_content))
                    latex_buffer = []
                    in_latex_block = False
            elif in_latex_block:
                latex_buffer.append(line)
            else:
                # Regular text line - check if it needs symbol conversion
                if line.strip():
                    result.append(self.convert_symbols_in_text(line))
                else:
                    result.append(line)

        # Process any remaining LaTeX buffer
        if latex_buffer:
            result.append(self.convert_latex_content('\n'.join(latex_buffer)))

        return '\n'.join(result)

    def convert_latex_content(self, text: str) -> str:
        """Convert LaTeX content, handling Unicode symbols within it."""
        # Replace Unicode symbols with LaTeX commands in the LaTeX code
        for symbol, latex_cmd in self.symbol_map.items():
            # In LaTeX environments, we don't need the $ signs
            text = text.replace(symbol, latex_cmd)

        # Clean up any redundant $ signs that might have been in the original
        text = re.sub(r'\$\s*\$', '', text)

        return text

    def smart_math_wrap(self, text: str) -> str:
        """
        Intelligently wrap math symbols in $ signs, handling consecutive symbols
        and avoiding double-wrapping.
        """
        if not text:
            return text

        # Don't process if this looks like LaTeX code
        if self.is_latex_content(text):
            return text

        # Track positions of all symbols
        symbol_positions = []
        for symbol in self.symbol_map:
            pos = 0
            while pos < len(text):
                pos = text.find(symbol, pos)
                if pos == -1:
                    break
                symbol_positions.append((pos, pos + len(symbol), symbol))
                pos += len(symbol)

        if not symbol_positions:
            return text

        # Sort by position
        symbol_positions.sort(key=lambda x: x[0])

        # Group consecutive symbols
        groups = []
        if symbol_positions:
            current_group = [symbol_positions[0]]

            for i in range(1, len(symbol_positions)):
                prev_end = symbol_positions[i - 1][1]
                curr_start = symbol_positions[i][0]

                # Check if symbols are consecutive or have only spaces between them
                between_text = text[prev_end:curr_start]
                if between_text.strip() == '' and len(between_text) <= 3:
                    current_group.append(symbol_positions[i])
                else:
                    groups.append(current_group)
                    current_group = [symbol_positions[i]]

            groups.append(current_group)

        # Build result by replacing groups
        result = []
        last_pos = 0

        for group in groups:
            group_start = group[0][0]
            group_end = group[-1][1]

            # Add text before this group
            result.append(text[last_pos:group_start])

            # Convert the group
            group_text = text[group_start:group_end]
            for symbol, latex in self.symbol_map.items():
                group_text = group_text.replace(symbol, latex)

            # Wrap in math mode
            result.append(f'${group_text}$')
            last_pos = group_end

        # Add remaining text
        result.append(text[last_pos:])

        return ''.join(result)

    def convert_symbols_in_text(self, text: str) -> str:
        """
        Convert symbols in regular text, handling math mode intelligently.
        """
        # Don't process LaTeX environments
        if self.is_latex_content(text):
            return self.convert_latex_content(text)

        # Split by existing math regions
        parts = []
        segments = []

        # Find all math regions
        math_patterns = [
            (r'\$[^\$]+\$', 'inline'),  # $...$
            (r'\$\$[^\$]+\$\$', 'display'),  # $$...$$
            (r'\\\[[^\]]*\\\]', 'display'),  # \[...\]
            (r'\\begin\{equation\}.*?\\end\{equation\}', 'equation'),
            (r'\\begin\{align\}.*?\\end\{align\}', 'align'),
            (r'\\begin\{gather\}.*?\\end\{gather\}', 'gather'),
        ]

        protected_regions = []
        for pattern, mode in math_patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                protected_regions.append((match.start(), match.end()))

        # Sort and merge overlapping regions
        protected_regions.sort()
        merged_regions = []
        for start, end in protected_regions:
            if merged_regions and start <= merged_regions[-1][1]:
                merged_regions[-1] = (merged_regions[-1][0], max(end, merged_regions[-1][1]))
            else:
                merged_regions.append((start, end))

        # Process text segments
        result = []
        last_pos = 0

        for start, end in merged_regions:
            # Process text before protected region
            if start > last_pos:
                segment = text[last_pos:start]
                result.append(self.smart_math_wrap(segment))

            # Keep protected region as-is
            result.append(text[start:end])
            last_pos = end

        # Process remaining text
        if last_pos < len(text):
            segment = text[last_pos:]
            result.append(self.smart_math_wrap(segment))

        return ''.join(result)

    def clean_header(self, text: str) -> str:
        """Clean header text by removing leading numbers."""
        cleaned = re.sub(r'^(\d+\.)+\s*', '', text.strip())
        # Convert symbols in headers
        return self.convert_symbols_in_text(cleaned)

    def detect_and_convert_table(self, text: str) -> str:
        """Detect ASCII tables and convert them to LaTeX format."""
        lines = text.split('\n')
        result = []
        i = 0

        while i < len(lines):
            if self.is_table_start(lines, i):
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

        for i in range(start_index, min(start_index + 5, len(lines))):
            line = lines[i].strip()
            if '|' in line or re.match(r'^[=\-]+$', line):
                for j in range(i, min(i + 3, len(lines))):
                    if '|' in lines[j]:
                        return True
        return False

    def extract_and_convert_table(self, lines, start_index):
        """Extract a table from lines and convert to LaTeX."""
        table_lines = []
        end_index = start_index

        for i in range(start_index, len(lines)):
            line = lines[i].strip()

            if not line and not table_lines:
                continue

            if '|' in line or re.match(r'^[=\-]+$', line) or (table_lines and not line):
                if not line and table_lines:
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
                if table_lines:
                    end_index = i - 1
                    break

        if not table_lines:
            return None, start_index

        rows = []
        headers = None

        for line in table_lines:
            line = line.strip()

            if re.match(r'^[=\-]+$', line):
                continue

            if '|' in line:
                cells = [cell.strip() for cell in line.split('|')]
                while cells and not cells[0]:
                    cells.pop(0)
                while cells and not cells[-1]:
                    cells.pop()

                if cells:
                    if headers is None:
                        headers = cells
                    else:
                        rows.append(cells)

        if not headers:
            return None, start_index

        latex = self.format_latex_table(headers, rows)
        return latex, end_index

    def format_latex_table(self, headers, rows):
        """Format a table as LaTeX with proper math mode handling."""
        num_cols = len(headers)

        latex = "\\begin{table}[h!]\n"
        latex += "\\centering\n"
        latex += "\\begin{tabular}{" + "|c" * num_cols + "|}\n"
        latex += "\\hline\n"

        # Process headers with smart symbol conversion
        header_cells = []
        for h in headers:
            if any(symbol in h for symbol in self.symbol_map):
                converted = self.convert_symbols_for_table_cell(h)
                header_cells.append(converted)
            else:
                header_cells.append(h)

        latex += " & ".join(header_cells) + " \\\\\n"
        latex += "\\hline\n"

        # Process rows
        for row in rows:
            while len(row) < num_cols:
                row.append('')

            row_cells = []
            for cell in row[:num_cols]:
                if any(symbol in cell for symbol in self.symbol_map):
                    converted = self.convert_symbols_for_table_cell(cell)
                    row_cells.append(converted)
                else:
                    row_cells.append(cell)

            latex += " & ".join(row_cells) + " \\\\\n"

        latex += "\\hline\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

        return latex

    def convert_symbols_for_table_cell(self, text: str) -> str:
        """Convert symbols in a table cell, grouping adjacent symbols."""
        if not text:
            return text

        # First replace all symbols with their LaTeX equivalents
        for symbol, latex in self.symbol_map.items():
            text = text.replace(symbol, f'SYMBOL_{latex}_SYMBOL')

        # Now wrap in math mode
        text = re.sub(r'SYMBOL_(.*?)_SYMBOL', r'$\1$', text)

        # Merge adjacent math modes
        text = re.sub(r'\$\s*\$', ' ', text)

        return text

    def convert(self):
        """Main conversion method."""
        with open(self.notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        latex_content = []
        chapter_title = None

        for cell in notebook['cells']:
            if cell['cell_type'] == 'markdown':
                content = self.process_markdown(cell)
                if content:
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

        output = f"% Generated from {self.notebook_path.name}\n\n"
        if chapter_title:
            output += f"\\chapter{{{chapter_title}}}\n\n"
        output += '\n\n'.join(latex_content)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            f.write(output)

        print(f"✔ Converted: {self.notebook_path.name} → {self.output_path}")

    def process_markdown(self, cell):
        """Convert markdown cell to LaTeX."""
        text = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if not text.strip():
            return ''

        # Check if this is LaTeX content that should be rendered directly
        if self.is_latex_content(text):
            return self.process_latex_output(text)

        # First detect and convert tables
        text = self.detect_and_convert_table(text)

        # Process headers
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

        # Process inline code BEFORE converting symbols
        text = re.sub(r'`([^`]+)`', r'\\pyinline{\1}', text)

        # Process math environments BEFORE symbol conversion
        # Display math
        text = re.sub(r'\$\$(.+?)\$\$',
                      lambda m: '\\[' + self.convert_math_content(m.group(1)) + '\\]',
                      text, flags=re.DOTALL)

        # Inline math
        text = re.sub(r'(?<!\$)\$(?!\$)([^\$]+)\$(?!\$)',
                      lambda m: '$' + self.convert_math_content(m.group(1)) + '$',
                      text)

        # Now convert symbols in the remaining text
        text = self.convert_symbols_in_text(text)

        # Text formatting (do after symbol conversion to avoid issues)
        text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
        text = re.sub(r'(?<!\*)\*([^\*]+)\*(?!\*)', r'\\textit{\1}', text)

        # Quotes
        text = re.sub(r'^> (.+)$', lambda m: self.format_quote(m.group(1)), text, flags=re.MULTILINE)

        # Lists
        text = self.process_lists(text)

        # Links
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\\href{\2}{\1}', text)

        return text

    def convert_math_content(self, math_text: str) -> str:
        """Convert symbols within math mode (no dollar signs needed)."""
        for symbol, latex in self.symbol_map.items():
            math_text = math_text.replace(symbol, latex)
        return math_text

    def format_quote(self, text):
        """Format block quote with proper symbol handling."""
        text = self.convert_symbols_in_text(text)

        if ' - ' in text:
            parts = text.split(' - ', 1)
            return f'\\begin{{quote}}\n\\textit{{{parts[0].strip()}}}\n\\hfill --- {parts[1].strip()}\n\\end{{quote}}'
        return f'\\begin{{quote}}\n\\textit{{{text}}}\n\\end{{quote}}'

    def process_lists(self, text):
        """Convert markdown lists to LaTeX with proper symbol handling."""
        lines = text.split('\n')
        result = []
        in_list = False
        list_type = None
        indent_stack = []

        for line in lines:
            indent = len(line) - len(line.lstrip())

            unordered_match = re.match(r'^(\s*)[-*+] (.+)$', line)
            ordered_match = re.match(r'^(\s*)\d+\. (.+)$', line)

            if unordered_match:
                current_indent = len(unordered_match.group(1))
                item_content = unordered_match.group(2)
                item_content = self.convert_symbols_in_text(item_content)

                if not in_list:
                    result.append('\\begin{itemize}')
                    in_list = True
                    list_type = 'itemize'
                    indent_stack = [current_indent]

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
                item_content = self.convert_symbols_in_text(item_content)

                if not in_list:
                    result.append('\\begin{enumerate}')
                    in_list = True
                    list_type = 'enumerate'
                    indent_stack = [current_indent]

                if current_indent > indent_stack[-1]:
                    result.append('\\begin{enumerate}')
                    indent_stack.append(current_indent)
                elif current_indent < indent_stack[-1]:
                    while indent_stack and current_indent < indent_stack[-1]:
                        result.append('\\end{enumerate}')
                        indent_stack.pop()

                result.append(f'\\item {item_content}')

            else:
                if in_list and line.strip() and indent > 0:
                    result.append(self.convert_symbols_in_text(line.strip()))
                elif in_list and line.strip() == '':
                    result.append('')
                elif in_list:
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

        if in_list:
            while indent_stack:
                if list_type == 'enumerate':
                    result.append('\\end{enumerate}')
                else:
                    result.append('\\end{itemize}')
                indent_stack.pop()

        return '\n'.join(result)

    def process_code(self, cell):
        """Convert code cell to LaTeX."""
        source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        if not source.strip():
            return ''

        # Don't convert symbols in actual code
        latex = "\\begin{pythoncode}\n"
        latex += source.strip()
        latex += "\n\\end{pythoncode}"

        outputs = self.extract_output(cell)
        if outputs:
            # Check if output contains LaTeX content
            if self.is_latex_content(outputs):
                # Process as LaTeX content
                processed_output = self.process_latex_output(outputs)
                latex += "\n" + processed_output
            elif self.looks_like_table_output(outputs):
                # Try to convert as table
                converted_output = self.detect_and_convert_table(outputs)
                if '\\begin{table}' in converted_output:
                    latex += "\n" + converted_output
                else:
                    # Regular verbatim output
                    latex += "\n\\begin{codeoutput}\n"
                    latex += "\\begin{verbatim}\n"
                    latex += outputs
                    latex += "\n\\end{verbatim}\n"
                    latex += "\\end{codeoutput}"
            else:
                # Regular output - keep in verbatim
                latex += "\n\\begin{codeoutput}\n"
                latex += "\\begin{verbatim}\n"
                latex += outputs
                latex += "\n\\end{verbatim}\n"
                latex += "\\end{codeoutput}"

        return latex

    def looks_like_table_output(self, text):
        """Check if the output looks like a table."""
        lines = text.split('\n')
        pipe_count = sum(1 for line in lines if '|' in line)
        separator_count = sum(1 for line in lines if re.match(r'^[=\-]+$', line.strip()))
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
                if 'traceback' in output:
                    for line in output['traceback']:
                        clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                        output_lines.append(clean_line)

        return ''.join(output_lines).strip()


if __name__ == '__main__':
    # Convert the notebook
    JupyterToLatexConverter('biljeznice/wittgenstein.ipynb', 'knjiga/wittgenstein.tex')
    JupyterToLatexConverter('biljeznice/gentzen.ipynb', 'knjiga/gentzen.tex')
    JupyterToLatexConverter('biljeznice/tarski.ipynb', 'knjiga/tarski.tex')
    JupyterToLatexConverter('biljeznice/turing.ipynb', 'knjiga/turing.tex')