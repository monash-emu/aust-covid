

class TexDoc:
    def __init__(self, path, doc_name):
        self.content = {}
        self.packages = []
        self.path = path
        self.doc_name = f'{doc_name}.tex'
    
    def add_line(self, line, section):
        if section in self.content:
            self.content[section].append(line)
        else:
            self.content[section] = [line]

    def write_line(self, line):
        self.file.write(f'{line}\n')

    def start_doc(self, bib_filename, title):
        self.write_line('\\documentclass{article}')
        for line in self.packages:
            self.write_line(line)
        self.write_line(f'\\addbibresource{{{bib_filename}.bib}}')
        self.write_line(f'\\title{{{title}}}')
        self.write_line('\\begin{document}')
        self.write_line('\maketitle')

    def populate_doc(self, content):
        for section in content.keys():
            self.write_line(f'\n\\section{{{section}}}')
            for element in content[section]:
                self.file.write(element)
            self.file.write('\n')
        
    def finish_doc(self):
        self.write_line('\\printbibliography')
        self.write_line('\\end{document}')
        
    def write_tex(self, bib_name, title):
        with open(self.path /self.doc_name, 'w') as self.file:
            self.start_doc(bib_name, title)
            self.populate_doc(self.content)
            self.finish_doc()
            
class StandardTexDoc(TexDoc):
    def __init__(self, path, doc_name):
        super().__init__(path, doc_name)
        core_packages = [
            '\\usepackage{biblatex}',
            '\\usepackage{hyperref}',
        ]    
        self.packages += core_packages
