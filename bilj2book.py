#!/usr/bin/env python3
"""
Optimized Jupyter Notebook to LaTeX Converter
Focuses on clean architecture, performance, and proper math symbol conversion.
Enhanced with quotation recognition and title number removal.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class MathSymbolRegistry:
    """Centralized registry for math symbol mappings with caching."""

    def __init__(self):
        # Unicode to LaTeX mappings organized by category
        self.LOGIC_SYMBOLS = {
            '⊥': r'\bot', '⊤': r'\top', '¬': r'\lnot',
            '∧': r'\land', '∨': r'\lor',
            '→': r'\rightarrow', '←': r'\leftarrow', '↔': r'\leftrightarrow',
            '⇒': r'\Rightarrow', '⇐': r'\Leftarrow', '⇔': r'\Leftrightarrow',
            '∀': r'\forall', '∃': r'\exists', '∄': r'\nexists',
            '⊢': r'\vdash', '⊨': r'\models',
        }

        self.SET_SYMBOLS = {
            '∈': r'\in', '∉': r'\notin', '∅': r'\emptyset',
            '⊂': r'\subset', '⊆': r'\subseteq', '⊇': r'\supseteq',
            '∪': r'\cup', '∩': r'\cap',
            '⊕': r'\oplus', '⊗': r'\otimes',
        }

        self.OPERATOR_SYMBOLS = {
            '×': r'\times', '÷': r'\div', '±': r'\pm',
            '≠': r'\neq', '≡': r'\equiv', '≈': r'\approx',
            '≤': r'\leq', '≥': r'\geq', '≪': r'\ll', '≫': r'\gg',
            '∞': r'\infty', '∂': r'\partial', '∇': r'\nabla',
            '∑': r'\sum', '∏': r'\prod', '∫': r'\int',
        }

        self.GREEK_LETTERS = {
            'α': r'\alpha', 'β': r'\beta', 'γ': r'\gamma', 'δ': r'\delta',
            'ε': r'\varepsilon', 'θ': r'\theta', 'λ': r'\lambda', 'μ': r'\mu',
            'π': r'\pi', 'σ': r'\sigma', 'φ': r'\phi', 'ψ': r'\psi', 'ω': r'\omega',
            'Γ': r'\Gamma', 'Δ': r'\Delta', 'Λ': r'\Lambda', 'Σ': r'\Sigma',
            'Φ': r'\Phi', 'Ψ': r'\Psi', 'Ω': r'\Omega',
        }

        self.COMPOUND_SYMBOLS = {
            'ℵ₀': r'\aleph_0', 'ℵ₁': r'\aleph_1', 'ℵ₂': r'\aleph_2',
            'ℵ': r'\aleph', '2^ℵ₀': r'2^{\aleph_0}',
        }

        self.SUBSCRIPTS = {
            '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
            '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9'
        }

        self.SUPERSCRIPTS = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
            '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
        }

        # Cache combined symbols
        self._all_symbols = None
        self._spacing_rules = None

    def all_symbols(self) -> Dict[str, str]:
        """Get all symbols combined (cached)."""
        if self._all_symbols is None:
            self._all_symbols = {}
            self._all_symbols.update(self.LOGIC_SYMBOLS)
            self._all_symbols.update(self.SET_SYMBOLS)
            self._all_symbols.update(self.OPERATOR_SYMBOLS)
            self._all_symbols.update(self.GREEK_LETTERS)
        return self._all_symbols

    def get_spacing_rules(self) -> Dict[str, Set[str]]:
        """Get spacing rules for different symbol types."""
        if self._spacing_rules is None:
            self._spacing_rules = {
                'unary_prefix': {r'\lnot'},
                'binary_ops': {
                    r'\land', r'\lor', r'\rightarrow', r'\leftarrow', r'\leftrightarrow',
                    r'\Rightarrow', r'\Leftarrow', r'\Leftrightarrow', r'\models', r'\vdash',
                    r'\equiv', r'\approx', r'\leq', r'\geq', r'\ll', r'\gg', r'\neq',
                    r'\times', r'\div', r'\oplus', r'\otimes', r'\pm',
                    r'\cup', r'\cap', r'\subset', r'\subseteq', r'\supseteq', r'\in', r'\notin'
                }
            }
        return self._spacing_rules


class ProtectedContent:
    """Manages protection and restoration of content during processing."""

    def __init__(self):
        self.protected_items: List[str] = []
        self.counter = 0

    def protect(self, content: str, tag: str = "PROTECTED") -> str:
        """Store content and return placeholder."""
        self.protected_items.append(content)
        placeholder = f"__{tag}_{self.counter}__"
        self.counter += 1
        return placeholder

    def restore_all(self, text: str) -> str:
        """Restore all protected content."""
        for i, content in enumerate(self.protected_items):
            for tag in ["PROTECTED", "MATH", "QUOTE", "DOLLAR", "BLOCKQUOTE"]:
                text = text.replace(f"__{tag}_{i}__", content)
        return text


class OptimizedMathProcessor:
    """Optimized math processing with precompiled patterns."""

    def __init__(self):
        self.symbols = MathSymbolRegistry()
        self._compile_patterns()

    def _compile_patterns(self):
        """Precompile regex patterns for performance."""
        # Math environment patterns
        self.display_math_pattern = re.compile(r'\\\[.*?\\\]|\$\$.*?\$\$', re.DOTALL)
        self.inline_math_pattern = re.compile(r'\$(?!\$)[^$]+?\$')

        # Quote patterns - poboljšano prepoznavanje citata
        self.double_quote_pattern = re.compile(r'"[^"\n]*"')
        self.single_quote_pattern = re.compile(r"'[^'\n]*'")
        self.latex_quote_pattern = re.compile(r'\\begin\{quote\}.*?\\end\{quote\}', re.DOTALL)
        self.latex_quotation_pattern = re.compile(r'\\begin\{quotation\}.*?\\end\{quotation\}', re.DOTALL)

        # Math detection pattern
        all_cmds = sorted({
            cmd.lstrip('\\') for cmd in
            list(self.symbols.all_symbols().values()) +
            list(self.symbols.COMPOUND_SYMBOLS.values())
        })
        self.cmd_pattern = '|'.join(map(re.escape, all_cmds))
        self.has_math_cmd = re.compile(rf'\\(?:{self.cmd_pattern})\b')

        # Formula patterns
        self.paren_formula = re.compile(
            rf'(?<!\$)\((?P<inner>(?:[^()$]|\\(?:{self.cmd_pattern}))+?)\)(?!\$)'
        )

        # Cleanup patterns
        self.empty_math = re.compile(r'\$\s*\$')
        self.adjacent_math = re.compile(r'\$([^$]+)\$\s*\$([^$]+)\$')
        self.nested_dollar_paren = re.compile(r'\$\(\s*\$([^$]+)\$\s*\)\$')

    def process(self, text: str) -> str:
        """Main entry point for math processing with optimized flow."""
        if not text:
            return text

        protector = ProtectedContent()

        # Phase 1: Protect existing math environments, quotes and LaTeX quotations
        text = self._protect_existing_math(text, protector)

        # Phase 2: Convert symbols (only outside protected areas)
        text = self._convert_symbols(text)

        # Phase 3: Wrap formulas in math mode (smart detection)
        text = self._wrap_formulas(text)

        # Phase 4: Clean up and normalize
        text = self._cleanup_math(text)

        # Phase 5: Restore protected content
        text = protector.restore_all(text)

        return text

    def _protect_existing_math(self, text: str, protector: ProtectedContent) -> str:
        """Protect existing math environments, quoted content and LaTeX quotations."""
        # Protect LaTeX quotations first
        text = self.latex_quote_pattern.sub(
            lambda m: protector.protect(m.group(0), "BLOCKQUOTE"), text
        )
        text = self.latex_quotation_pattern.sub(
            lambda m: protector.protect(m.group(0), "BLOCKQUOTE"), text
        )

        # Protect display math
        text = self.display_math_pattern.sub(
            lambda m: protector.protect(m.group(0), "MATH"), text
        )

        # Protect inline math
        text = self.inline_math_pattern.sub(
            lambda m: protector.protect(m.group(0), "MATH"), text
        )

        # Protect quoted content (examples in quotes should stay verbatim)
        text = self.double_quote_pattern.sub(
            lambda m: protector.protect(m.group(0), "QUOTE"), text
        )
        text = self.single_quote_pattern.sub(
            lambda m: protector.protect(m.group(0), "QUOTE"), text
        )

        return text

    def _convert_symbols(self, text: str) -> str:
        """Convert Unicode symbols to LaTeX with intelligent spacing."""
        # Process compound symbols first
        for unicode_sym, latex_cmd in self.symbols.COMPOUND_SYMBOLS.items():
            text = text.replace(unicode_sym, latex_cmd)

        # Get spacing rules
        rules = self.symbols.get_spacing_rules()

        # Process individual symbols with spacing
        for unicode_sym, latex_cmd in self.symbols.all_symbols().items():
            # Determine spacing
            if latex_cmd in rules['unary_prefix']:
                replacement = latex_cmd + ' '
            elif latex_cmd in rules['binary_ops']:
                replacement = ' ' + latex_cmd + ' '
            else:
                replacement = latex_cmd

            # Replace symbol
            text = text.replace(unicode_sym, replacement)

        # Handle subscripts and superscripts
        for sub, num in self.symbols.SUBSCRIPTS.items():
            text = text.replace(sub, f'_{num}')
        for sup, num in self.symbols.SUPERSCRIPTS.items():
            text = text.replace(sup, f'^{num}')

        # Normalize multiple spaces
        text = re.sub(r'[ \t]{2,}', ' ', text)

        return text

    def _wrap_formulas(self, text: str) -> str:
        """Intelligently wrap formulas in $ delimiters."""
        # Temporarily store already wrapped segments
        dollar_segments = []

        def save_dollar(m):
            dollar_segments.append(m.group(0))
            return f"__DOLLAR_{len(dollar_segments) - 1}__"

        # Save existing $ segments to avoid double wrapping
        text = self.inline_math_pattern.sub(save_dollar, text)

        # Wrap parenthesized formulas containing math commands
        def wrap_paren(m):
            inner = m.group('inner')
            if self.has_math_cmd.search(inner):
                # Clean any stray $ inside
                inner = inner.replace('$', '').strip()
                return f'$({inner})$'
            return m.group(0)

        text = self.paren_formula.sub(wrap_paren, text)

        # Wrap standalone math command sequences
        # More conservative approach: only wrap clear formula runs
        math_run = re.compile(
            rf'(?<![A-Za-z$\\])({self.cmd_pattern})(?:\s+\w+|\s*[_^{{}}\d]+)*(?![A-Za-z$\\])'
        )

        def wrap_run(m):
            content = m.group(0).strip()
            # Don't wrap if it's just a lone operator
            if re.search(r'[A-Za-z0-9]', content) or len(content) > 3:
                return f'${content}$'
            return m.group(0)

        text = math_run.sub(wrap_run, text)

        # Restore saved dollar segments
        for i, content in enumerate(dollar_segments):
            text = text.replace(f"__DOLLAR_{i}__", content)

        return text

    def _cleanup_math(self, text: str) -> str:
        """Clean up math delimiters and spacing."""
        # Remove empty math
        text = self.empty_math.sub(' ', text)

        # Merge adjacent inline math
        text = self.adjacent_math.sub(r'$\1 \2$', text)

        # Fix nested dollar parentheses
        text = self.nested_dollar_paren.sub(r'$(\1)$', text)

        # Fix spacing around math
        text = re.sub(r'\$\s+\(', r'$(', text)
        text = re.sub(r'\)\s+\$', r')$', text)

        # Ensure space between math and text
        text = re.sub(r'([A-Za-zČčĆćĐđŠšŽž])\$\(', r'\1 $(', text)
        text = re.sub(r'\)\$([A-Za-zČčĆćĐđŠšŽž])', r')$ \1', text)

        # Space before punctuation
        text = re.sub(r'\)\$([?!])', r')$ \1', text)

        return text


class OptimizedMarkdownProcessor:
    """Optimized markdown to LaTeX conversion with precompiled patterns."""

    def __init__(self, math_processor: OptimizedMathProcessor):
        self.math_processor = math_processor
        self._compile_patterns()

    def _compile_patterns(self):
        """Precompile regex patterns."""
        self.header_empty = re.compile(r'^#{1,6}\s*$', re.MULTILINE)
        # Poboljšan pattern za uklanjanje brojeva s početka naslova
        self.header_content = re.compile(r'^(#{1,6})\s+(?:\d+\.?\d*\.?\s*)?(.+)$', re.MULTILINE)
        self.display_math = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
        self.bold = re.compile(r'\*\*([^\*]+?)\*\*')
        self.italic = re.compile(r'(?<!\*)\*([^\*]+)\*(?!\*)')
        self.inline_code = re.compile(r'`([^`]+)`')
        self.link = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.list_item = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.+)$')
        # Novi pattern za markdown blockquote
        self.blockquote = re.compile(r'^>\s+(.+)$', re.MULTILINE)
        self.multiline_blockquote = re.compile(r'^((?:>\s*.+\n?)+)', re.MULTILINE)

    def process(self, content: str) -> str:
        """Convert markdown to LaTeX."""
        if not content.strip():
            return ''

        # Process in optimized order
        content = self._convert_blockquotes(content)  # Novo - obradi citate
        content = self._convert_headers(content)
        content = self._convert_display_math(content)
        content = self._convert_emphasis(content)
        content = self._convert_code_blocks(content)
        content = self._convert_lists(content)
        content = self._convert_links(content)

        # Process math symbols last
        content = self.math_processor.process(content)

        return content

    def _convert_blockquotes(self, text: str) -> str:
        """Convert markdown blockquotes to LaTeX quote environment."""

        def replace_blockquote(match):
            # Izvuci sve linije koje počinju sa >
            lines = match.group(1).split('\n')
            cleaned_lines = []
            for line in lines:
                if line.startswith('>'):
                    # Ukloni > i razmake s početka
                    cleaned_lines.append(line.lstrip('>').strip())

            quote_content = '\n'.join(cleaned_lines)
            return f'\\begin{{quote}}\n{quote_content}\n\\end{{quote}}'

        # Obradi višelinijske blockquote
        text = self.multiline_blockquote.sub(replace_blockquote, text)

        return text

    def _convert_headers(self, text: str) -> str:
        """Convert markdown headers to LaTeX sections, removing leading numbers."""
        # Remove empty headers
        text = self.header_empty.sub('', text)

        # Convert headers with content, removing numbers
        def replace_header(match):
            level = len(match.group(1))
            title = match.group(2).strip()

            # Dodatno uklanjanje brojeva ako nisu uhvaćeni regex-om
            # Uklanja pattern poput "1.", "1.1", "1.1.1", itd. s početka
            title = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', title).strip()

            commands = {
                1: 'section', 2: 'subsection', 3: 'subsubsection',
                4: 'paragraph', 5: 'subparagraph', 6: 'subparagraph'
            }

            return f'\\{commands.get(level, "paragraph")}{{{title}}}'

        return self.header_content.sub(replace_header, text)

    def _convert_display_math(self, text: str) -> str:
        """Convert display math delimiters."""
        return self.display_math.sub(r'\\[\1\\]', text)

    def _convert_emphasis(self, text: str) -> str:
        """Convert bold and italic markdown."""
        text = self.bold.sub(r'\\textbf{\1}', text)
        text = self.italic.sub(r'\\textit{\1}', text)
        return text

    def _convert_code_blocks(self, text: str) -> str:
        """Convert inline code to texttt."""

        def escape_texttt(code):
            """Escape special LaTeX characters."""
            escapes = [
                ('\\', r'\textbackslash '),
                ('{', r'\{'), ('}', r'\}'),
                ('_', r'\_'), ('^', r'\^{}'),
                ('%', r'\%'), ('&', r'\&'),
                ('#', r'\#'), ('$', r'\$'),
            ]
            for old, new in escapes:
                code = code.replace(old, new)
            return code

        return self.inline_code.sub(
            lambda m: f'\\texttt{{{escape_texttt(m.group(1))}}}', text
        )

    def _convert_lists(self, text: str) -> str:
        """Convert markdown lists to LaTeX with optimized processing."""
        lines = text.split('\n')
        result = []
        list_stack = []

        for line in lines:
            match = self.list_item.match(line)

            if match:
                indent = len(match.group(1)) // 2
                marker = match.group(2)
                content = match.group(3)
                list_type = 'enumerate' if marker[0].isdigit() else 'itemize'

                # Adjust list depth
                while len(list_stack) > indent + 1:
                    result.append(f'\\end{{{list_stack.pop()}}}')

                # Start new list if needed
                if len(list_stack) <= indent:
                    list_stack.append(list_type)
                    result.append(f'\\begin{{{list_type}}}')

                result.append(f'\\item {content}')
            else:
                # Close lists if needed
                if list_stack and line.strip() and not line.startswith(' '):
                    while list_stack:
                        result.append(f'\\end{{{list_stack.pop()}}}')
                result.append(line)

        # Close remaining lists
        while list_stack:
            result.append(f'\\end{{{list_stack.pop()}}}')

        return '\n'.join(result)

    def _convert_links(self, text: str) -> str:
        """Convert markdown links to LaTeX href."""
        return self.link.sub(r'\\href{\2}{\1}', text)


class CodeProcessor:
    """Handles code cell conversion with improved output processing."""

    # Precompile ANSI escape pattern
    ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')

    @classmethod
    def process(cls, cell: dict) -> str:
        """Convert code cell to LaTeX with minted."""
        source = ''.join(cell.get('source', []))
        if not source.strip():
            return ''

        # Create minted code block
        latex_parts = [
            "\\begin{minted}[bgcolor=bg,frame=lines,framesep=2mm,",
            "baselinestretch=1.2,fontsize=\\footnotesize]{python}",
            source.strip(),
            "\\end{minted}"
        ]

        # Add output if present
        output = cls._extract_output(cell)
        if output:
            latex_parts.extend([
                "",
                "\\begin{tcolorbox}[title=Output,colback=gray!5!white,colframe=gray!75!black]",
                "\\begin{verbatim}",
                output,
                "\\end{verbatim}",
                "\\end{tcolorbox}"
            ])

        return '\n'.join(latex_parts)

    @classmethod
    def _extract_output(cls, cell: dict) -> str:
        """Extract and clean text output from code cell."""
        if 'outputs' not in cell:
            return ''

        output_lines = []

        for output in cell['outputs']:
            output_type = output.get('output_type')

            if output_type == 'stream':
                text = output.get('text', [])
                output_lines.extend(text if isinstance(text, list) else [text])

            elif output_type == 'execute_result':
                data = output.get('data', {})
                if 'text/plain' in data:
                    text = data['text/plain']
                    output_lines.extend(text if isinstance(text, list) else [text])

            elif output_type == 'error':
                for line in output.get('traceback', []):
                    # Remove ANSI codes
                    clean = cls.ANSI_PATTERN.sub('', line)
                    output_lines.append(clean)

        return ''.join(output_lines).strip()


class JupyterToLatexConverter:
    """Main converter class with improved error handling and batch processing."""

    LATEX_HEADER = """% Generated from {filename}
% Requirements in preamble:
% \\usepackage{{amsmath,amssymb}}
% \\usepackage{{minted}}
% \\usepackage{{tcolorbox}}
% \\usepackage{{hyperref}}
% \\usepackage{{csquotes}}  % Za citate
% \\definecolor{{bg}}{{rgb}}{{0.95,0.95,0.95}}
% Compile with: pdflatex -shell-escape yourfile.tex"""

    def __init__(self):
        self.math_processor = OptimizedMathProcessor()
        self.markdown_processor = OptimizedMarkdownProcessor(self.math_processor)
        self.code_processor = CodeProcessor()

    def convert_notebook(self, input_path: Path, output_path: Path) -> None:
        """Convert a single notebook to LaTeX with error handling."""
        logger.info(f"Converting {input_path.name}...")

        try:
            # Load notebook
            with open(input_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)

            # Process cells
            latex_parts = [self.LATEX_HEADER.format(filename=input_path.name)]

            for i, cell in enumerate(notebook.get('cells', [])):
                try:
                    content = self._process_cell(cell)
                    if content:
                        latex_parts.append(content)
                except Exception as e:
                    logger.warning(f"Failed to process cell {i} in {input_path.name}: {e}")
                    continue

            # Write output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(latex_parts))

            logger.info(f"✓ Successfully converted: {input_path.name}")

        except Exception as e:
            logger.error(f"✗ Failed to convert {input_path.name}: {e}")
            raise

    def _process_cell(self, cell: dict) -> Optional[str]:
        """Process a single cell based on its type."""
        cell_type = cell.get('cell_type', '')

        if cell_type == 'markdown':
            source = ''.join(cell.get('source', []))
            return self.markdown_processor.process(source)

        elif cell_type == 'code':
            return self.code_processor.process(cell)

        return None

    def batch_convert(self, conversions: List[Tuple[str, str]],
                      continue_on_error: bool = True) -> Tuple[int, List[Tuple[str, str]]]:
        """Batch convert multiple notebooks with progress tracking."""
        successful = 0
        failed = []

        total = len(conversions)
        for i, (input_path, output_path) in enumerate(conversions, 1):
            logger.info(f"[{i}/{total}] Processing {input_path}")

            try:
                self.convert_notebook(Path(input_path), Path(output_path))
                successful += 1
            except Exception as e:
                error_msg = str(e)
                failed.append((input_path, error_msg))

                if not continue_on_error:
                    logger.error(f"Stopping batch conversion due to error: {error_msg}")
                    break

        return successful, failed


def main():
    """Main entry point with improved configuration."""
    converter = JupyterToLatexConverter()

    # Define conversions
    notebooks = [
        ('biljeznice/wittgenstein.ipynb', 'knjiga/wittgenstein.tex'),
        ('biljeznice/gentzen.ipynb', 'knjiga/gentzen.tex'),
        ('biljeznice/tarski.ipynb', 'knjiga/tarski.tex'),
        ('biljeznice/turing.ipynb', 'knjiga/turing.tex'),
        ('biljeznice/cantor.ipynb', 'knjiga/cantor.tex'),
        ('biljeznice/pascal.ipynb', 'knjiga/pascal.tex'),
        ('biljeznice/bayes.ipynb', 'knjiga/bayes.tex'),
        ('biljeznice/goodman.ipynb', 'knjiga/goodman.tex'),
    ]

    # Perform batch conversion
    successful, failed = converter.batch_convert(notebooks, continue_on_error=True)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"Conversion Summary: {successful}/{len(notebooks)} successful")

    if failed:
        print(f"\nFailed conversions:")
        for path, error in failed:
            print(f"  - {path}: {error}")

    return 0 if not failed else 1


if __name__ == '__main__':

    main()