

class TexDoc:
    def __init__(self, path, doc_name, title, bib_filename):
        self.content = {}
        self.path = path
        self.doc_name = f'{doc_name}.tex'
        self.bib_filename = bib_filename
        self.title = title
        self.prepared = False

    def add_line(self, line, section, subsection=None):
        if section not in self.content:
            self.content[section] = {}
        if not subsection:
            if '' not in self.content[section]:
                self.content[section][''] = []
            self.content[section][''].append(line)
        else:
            if subsection not in self.content[section]:
                self.content[section][subsection] = []
            self.content[section][subsection].append(line)

        
    def prepare_doc(self):
        self.prepared = True

    def show_doc(self):
        self.prepare_doc()
        output = ''
        for section in self.content:
            output += f'SECTION: {section}\n'
            for line in self.content[section]:
                output += f'{line}\n'
            output += '\n'
        return output

    def write_doc(self):
        with open(self.path / self.doc_name, 'w') as doc_file:
            doc_file.write(self.emit_doc())
    
    def emit_doc(self):
        if not self.prepared:
            self.prepare_doc()
        final_text = ''
        for line in self.content['preamble']['']:
            final_text += f'{line}\n'
        for section in [k for k in self.content.keys() if k not in ['preamble', 'endings']]:
            final_text += f'\n\\section{{{section}}}\n'
            if '' in self.content[section]:
                for line in self.content[section]['']:
                    final_text += f'{line}\n'
            for subsection in [k for k in self.content[section].keys() if k != '']:
                final_text += f'\n\\subsection{{{subsection}}}\n'
                for line in self.content[section][subsection]:
                    final_text += f'{line}\n'
        for line in self.content['endings']['']:
            final_text += f'{line}\n'
        return final_text

    def include_figure(self, caption, filename, section, subsection=None):
        self.add_line('\\begin{figure}', section, subsection)
        self.add_line(f'\\caption{{{caption}}}', section, subsection)
        self.add_line(f'\\includegraphics[width=\\textwidth]{{{filename}}}', section, subsection)
        self.add_line('\\end{figure}', section, subsection)

    def include_table(self, table, section, subsection=None, widths=None, table_width=10.0, longtable=False):
        n_cols = table.shape[1] + 1
        ave_col_width = round(table_width / n_cols, 2)
        col_widths = widths if widths else [ave_col_width] * n_cols
        col_format_str = ' '.join([f'>{{\\raggedright\\arraybackslash}}p{{{width}cm}}' for width in col_widths])
        table_text = table.style.to_latex(
            column_format=col_format_str,
            hrules=True,
        )
        table_text = table_text.replace('{tabular}', '{longtable}') if longtable else table_text
        self.add_line(table_text, section, subsection=subsection)


class StandardTexDoc(TexDoc):
    def prepare_doc(self):
        self.prepared = True
        self.add_line('\\documentclass{article}', 'preamble')

        # Packages that don't require arguments
        standard_packages = [
            'hyperref',
            'biblatex',
            'graphicx',
            'longtable',
            'booktabs',
            'array',
        ]
        for package in standard_packages:
            self.add_line(f'\\usepackage{{{package}}}', 'preamble')

        self.add_line('\\graphicspath{ {./images/} }', 'preamble')
        self.add_line(f'\\addbibresource{{{self.bib_filename}.bib}}', 'preamble')
        self.add_line(f'\\title{{{self.title}}}', 'preamble')
        self.add_line('\\begin{document}', 'preamble')
        self.add_line('\maketitle', 'preamble')
        
        self.add_line('\\printbibliography', 'endings')
        self.add_line('\\end{document}', 'endings')
            