class TexDoc:
    def __init__(self, path, doc_name):
        self.content = []
        self.packages = []
        self.path = path
        self.doc_name = f'{doc_name}.tex'
    
    def add_line(self, line):
        self.content.append(line)
    
    def write_line(self, line):
        self.file.write(f'{line}\n')

    def start_doc(self, bib_filename, title):
        self.write_line('\\documentclass{article}')
        for line in self.packages:
            self.write_line(line)
        self.write_line(f'\\addbibresource{{{bib_filename}.bib}}')
        self.write_line(f'\\title{{{title}}}')
        self.write_line('\\begin{document}')

    def populate_doc(self, elements):
        for element in elements:
            self.file.write(element)
        self.file.write('\n')
        
    def finish_doc(self):
        self.write_line('\\printbibliography')
        self.write_line('\\end{document}')
        
    def write_tex(self):
        with open(self.path / 'supplement' /self.doc_name, 'w') as self.file:
            self.start_doc('austcovid', 'Supplemental Appendix')
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
        