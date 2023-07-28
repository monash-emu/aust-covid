

class TexDoc:
    def __init__(self, path, doc_name, title, bib_filename):
        self.content = {}
        self.path = path
        self.doc_name = f'{doc_name}.tex'
        self.bib_filename = bib_filename
        self.title = title
        self.prepared = False

    def add_line(self, line, section):
        if section in self.content:
            self.content[section].append(line)
        else:
            self.content[section] = [line]
        
    def prepare_doc(self):
        self.prepared = True

    def show_doc(self):
        self.prepare_doc()
        for section in self.content:
            print(section)
            for line in self.content[section]:
                print(f'\t{line}')
            print('\n')
        
    def write_doc(self, final_text):
        with open(self.path / self.doc_name, 'w') as doc_file:
            doc_file.write(final_text)
    
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
            final_text += '\n'
        for line in self.content['endings']:
            final_text += f'{line}\n'
        self.write_doc(final_text)
                

class StandardTexDoc(TexDoc):
    def prepare_doc(self):
        self.prepared = True
        self.add_line('\\documentclass{article}', 'preamble')
        self.add_line('\\usepackage{biblatex}', 'preamble')
        self.add_line('\\usepackage{hyperref}', 'preamble')
        self.add_line(f'\\addbibresource{{{self.bib_filename}.bib}}', 'preamble')
        self.add_line(f'\\title{{{self.title}}}', 'preamble')
        self.add_line('\\begin{document}', 'preamble')
        self.add_line('\maketitle', 'preamble')
        
        self.add_line('\\printbibliography', 'endings')
        self.add_line('\\end{document}', 'endings')
            