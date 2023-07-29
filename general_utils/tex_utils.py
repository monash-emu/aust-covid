

class TexDoc:
    def __init__(self, path, doc_name, title, bib_filename):
        self.content = {}
        self.path = path
        self.doc_name = f'{doc_name}.tex'
        self.bib_filename = bib_filename
        self.title = title
        self.prepared = False

    def add_line(self, section, line):
        if section in self.content:
            self.content[section].append(line)
        else:
            self.content[section] = [line]
        
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
        for line in self.content['preamble']:
            final_text += f'{line}\n'
        for section in [k for k in self.content.keys() if k not in ['preamble', 'endings']]:
            final_text += f'\n\\section{{{section}}}\n'
            for line in self.content[section]:
                final_text += f'{line}\n'
        for line in self.content['endings']:
            final_text += f'{line}\n'
        return final_text

    def include_figure(self, section, caption, filename):
        self.add_line(section, '\\begin{figure}')
        self.add_line(section, f'\\caption{{{caption}}}')
        self.add_line(section, f'\\includegraphics[width=\\textwidth]{{{filename}}}')
        self.add_line(section, '\\end{figure}')
                

class StandardTexDoc(TexDoc):
    def prepare_doc(self):
        self.prepared = True
        self.add_line('preamble', '\\documentclass{article}')
        self.add_line('preamble', '\\usepackage{biblatex}')
        self.add_line('preamble', '\\usepackage{hyperref}')
        self.add_line('preamble', '\\usepackage{graphicx}')
        self.add_line('preamble', '\\graphicspath{ {./images/} }')

        self.add_line('preamble', f'\\addbibresource{{{self.bib_filename}.bib}}')
        self.add_line('preamble', f'\\title{{{self.title}}}')
        self.add_line('preamble', '\\begin{document}')
        self.add_line('preamble', '\maketitle')
        
        self.add_line('endings', '\\printbibliography')
        self.add_line('endings', '\\end{document}')
            