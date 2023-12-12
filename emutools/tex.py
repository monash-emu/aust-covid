from typing import Union, List
from pathlib import Path
import pandas as pd
import numpy as np
import yaml as yml
from datetime import datetime
from abc import abstractmethod, ABC
from matplotlib.figure import Figure as MplFig
from matplotlib import pyplot as plt
from plotly.graph_objects import Figure as PlotlyFig


def get_tex_formatted_date(
    date: datetime,
) -> str:
    """Get TeX-formatted date from the date of a datetime object.

    Args:
        date: The date input

    Returns:
        Formatted string
    """
    date_of_month = date.strftime('%d').lstrip('0')
    special_cases = {
        '1': 'st', 
        '2': 'nd',
        '3': 'rd',
        '21': 'st',
        '22': 'nd',
        '23': 'rd',
        '31': 'st',
    }
    text_super = special_cases[date_of_month] if date_of_month in special_cases else 'th'
    return f'{date_of_month}\\textsuperscript{{{text_super}}}{date.strftime(" of %B %Y")}'


def remove_underscore_multiindexcol(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Remove underscores from multi-index columns
    (because TeX so often crashes with underscores in floats, such as tables).

    Args:
        df: Dataframe to modify

    Returns:
        Revised dataframe
    """
    for l in range(df.columns.nlevels):
        new_index = df.columns.levels[l].str.replace('_', ' ')
        df.columns = df.columns.set_levels(new_index, level=l)
    return df


def get_tex_longtable(
    table: pd.DataFrame, 
    col_format_str: str, 
    caption_str: str, 
    label_str: str,
) -> str:
    """Get TeX longtable code from dataframe.

    Args:
        table: The pandas table
        col_format_str: The previously created TeX column format request
        caption_str: The previously formatted TeX caption request
        label_str: The previously formatted TeX label request

    Returns:
        Completed TeX string for table
    """
    start_str = '\\begin{longtable}\n'
    table_text = table.style.to_latex(column_format=col_format_str, hrules=True)
    table_text = table_text.replace('\\begin{tabular}', '')
    table_text = table_text.replace('\\end{tabular}', '')
    end_str = '\\end{longtable}'
    return start_str + table_text + caption_str + label_str + end_str


def get_tex_table(
    table: pd.DataFrame, 
    col_format_str: str, 
    caption_str: str, 
    label_str: str,
) -> str:
    """Get TeX table code from dataframe.

    Args:
        table: The pandas table
        col_format_str: The previously created TeX column format request
        caption_str: The previously formatted TeX caption request
        label_str: The previously formatted TeX label request

    Returns:
        Completed TeX string for table
    """
    start_str = '\\begin{table}\n'
    table_text = table.style.to_latex(column_format=col_format_str, hrules=True)
    end_str = '\\end{table}'
    return start_str + table_text + caption_str + label_str + end_str


class TexDoc(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def add_line(self, line: str, section: str, subsection: str=''):
        pass

    @abstractmethod
    def prepare_doc(self):
        pass

    @abstractmethod
    def write_doc(self, order: list=[]):
        pass

    @abstractmethod
    def emit_doc(self, section_order: list=[]) -> str:
        pass

    @abstractmethod
    def include_figure(self, title: str, filename: str, filetype: str, fig_path: str, section: str, subsection: str='', caption: str='', fig_width: float=1.0):
        pass

    @abstractmethod
    def include_table(self, table: pd.DataFrame, section: str, subsection: str='', col_splits=None, table_width=14.0, longtable=False):
        pass

    @abstractmethod
    def save_content(self):
        pass

    @abstractmethod
    def load_content(self):
        pass


class DummyTexDoc(TexDoc):
    """Minimal concrete object for use where 
    functions require a document to write to,
    but writing not desired.

    Args:
        TexDoc: Inherits from abstract class
    """
    def add_line(self, line: str, section: str, subsection: str=''):
        pass

    def prepare_doc(self):
        pass

    def write_doc(self, order: list=[]):
        pass

    def emit_doc(self, section_order: list=[]) -> str:
        pass

    def include_figure(self, title: str, filename: str, filetype: str, fig_path: str, section: str, subsection: str='', caption: str='', fig_width: float=1.0):
        pass

    def include_table(self, table: pd.DataFrame, section: str, subsection: str='', col_splits=None, table_width=14.0, longtable=False):
        pass

    def save_content(self):
        pass

    def load_content(self):
        pass


class ConcreteTexDoc:
    def __init__(
        self, 
        path: Path, 
        doc_name: str, 
        title: str, 
        bib_filename: str,
        table_of_contents: bool=False,
    ):
        """Object for collating document elements and emitting of a TeX-formatted string.

        Args:
            path: Path for writing the output document
            doc_name: Filename for the document produced
            title: Title to go in the document
            bib_filename: Name of the bibliography file
            table_of_contents: Whether to include a table of contents
        """
        self.content = {}
        self.path = path
        self.doc_name = doc_name
        self.bib_filename = bib_filename
        self.title = title
        self.prepared = False
        self.standard_sections = ['preamble', 'endings']
        self.table_of_contents = table_of_contents

    def add_line(
        self, 
        line: str, 
        section: str, 
        subsection: str='',
    ):
        """Add a single line string to the appropriate section/subsection of the document
        (adding to the start/intro of the section before sub-sections start 
        if no subsection is requested).

        Args:
            line: The TeX line to write
            section: The heading of the section for the line to go into
            subsection: The heading of the subsection for the line to go into
        """
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
        """Placeholder method for overwriting in parent class.
        """
        self.prepared = True

    def write_doc(self, order: list=[]):
        """Write the compiled document string to disc.

        Args:
            order: Section order to pass through to emit_doc method
        """
        with open(self.path / f'{self.doc_name}.tex', 'w') as doc_file:
            doc_file.write(self.emit_doc(section_order=order))
    
    def emit_doc(self, section_order: list=[]) -> str:
        """Collate all the sections together into the big string to be outputted.

        Arguments:
            section_order: The order to write the document sections in

        Returns:
            The final text to write into the document
        """
        content_sections = sorted([s for s in self.content if s not in self.standard_sections])
        if section_order and sorted(section_order) != content_sections:
            msg = 'Sections requested are not those in the current contents'
            raise ValueError(msg)
        order = section_order if section_order else self.content.keys()

        if not self.prepared:
            self.prepare_doc()
        final_text = ''
        for line in self.content['preamble']['']:
            final_text += f'{line}\n'
        for section in [k for k in order if k not in self.standard_sections]:
            final_text += f'\n\\section{{{section}}} \\label{{{section.lower().replace(" ", "_")}}}\n'
            if '' in self.content[section]:
                for line in self.content[section]['']:
                    final_text += f'{line}\n'
            for subsection in [k for k in self.content[section].keys() if k != '']:
                final_text += f'\n\\subsection{{{subsection}}} \\label{{{subsection.lower().replace(" ", "_")}}}\n'
                for line in self.content[section][subsection]:
                    final_text += f'{line}\n'
        for line in self.content['endings']['']:
            final_text += f'{line}\n'
        return final_text

    def include_figure(
        self, 
        title: str, 
        filename: str, 
        filetype: str,
        fig_path: Path,
        section: str, 
        subsection: str='',
        caption: str='',
        fig_width: float=0.85,
    ):
        """Add a figure with standard formatting to the document.

        Args:
            title: Figure title
            filename: Filename for finding the image file
            filetype: File extension (determines TeX command to use to include the figure)
            fig_path: Path where the figure file can be found
            section: The heading of the section for the figure to go into
            subsection: The heading of the subsection for the figure to go into
            caption: Figure caption
            fig_width: Figure width relative to document width
        """
        if filetype == 'jpg':
            command = 'includegraphics'
        elif filetype == 'svg':
            command = 'includesvg'
        else:
            raise ValueError('File type for figure not supported yet')
        self.add_line('\\begin{figure}[H]', section, subsection)
        self.add_line(f'\\caption{{\\textbf{{{title}}} {caption}}}', section, subsection)
        self.add_line('\\begin{adjustbox}{center, max width=\paperwidth}', section, subsection)
        command_str = f'\\{command}[width={str(round(fig_width, 2))}\\paperwidth]{{./{fig_path}/{filename}.{filetype}}}'
        self.add_line(command_str, section, subsection)
        self.add_line('\\end{adjustbox}', section, subsection)
        self.add_line(f'\\label{{{filename}}}', section, subsection)
        self.add_line('\\end{figure}\n', section, subsection)

    def include_table(
        self, 
        table: pd.DataFrame, 
        name: str,
        title: str,
        section: str, 
        subsection: str='', 
        col_splits: Union[List[float], None]=None, 
        table_width: float=14.0, 
        longtable: bool=False,
        caption: str='',
    ):
        """Use a dataframe to add a table to the working document.

        Args:
            table: The table to be written
            name: Short name of table for label
            title: Title for table
            section: The heading of the section for the figure to go into
            subsection: The heading of the subsection for the figure to go into
            col_splits: Optional user request for columns widths if not evenly distributed
            table_width: Overall table width if widths not requested
            longtable: Whether to use the longtable module to span pages
        """
        n_cols = table.shape[1] + 1
        if not col_splits:
            splits = [round(1.0 / n_cols, 4)] * n_cols
        elif len(col_splits) != n_cols:
            raise ValueError('Wrong number of proportion column splits requested')
        else:
            splits = col_splits
        col_widths = [w * table_width for w in splits]
        col_str = ' '.join([f'>{{\\raggedright\\arraybackslash}}p{{{width}cm}}' for width in col_widths])
        label_str = f'\label{{{name}}}\n'
        caption_str = f'\caption{{\\textbf{{{title}}} {caption}}}\n'
        table_str = get_tex_longtable(table, col_str, caption_str, label_str) if longtable else get_tex_table(table, col_str, caption_str, label_str)
        self.add_line(table_str, section, subsection=subsection)

    def save_content(self):
        """Save the current document information as a simple string.
        """
        with open(self.path / f'{self.doc_name}.yml', 'w') as file:
            yml.dump(self.content, file)

    def load_content(self):
        """Read saved document information. 
        """
        with open(self.path / f'{self.doc_name}.yml', 'r') as file:
            self.content = yml.load(file, Loader=yml.FullLoader)


class StandardTexDoc(ConcreteTexDoc):
    def prepare_doc(self):
        """Add packages and text that standard documents need to include the other features.
        """
        self.prepared = True
        self.add_line('\\documentclass{article}', 'preamble')

        # Packages that don't require arguments
        standard_packages = [
            'hyperref',
            'graphicx',
            'longtable',
            'booktabs',
            'array',
            'svg',
            'adjustbox',
            'float',
        ]
        for package in standard_packages:
            self.add_line(f'\\usepackage{{{package}}}', 'preamble')
        self.add_line('\DeclareUnicodeCharacter{2212}{-}', 'preamble')  # SVG compilation often crashes without this
        self.add_line('\\usepackage[sorting=none, style=science]{biblatex}', 'preamble')

        self.add_line(r'\usepackage[a4paper, total={15cm, 20cm}]{geometry}', 'preamble')
        self.add_line(r'\usepackage[labelfont=bf,it]{caption}', 'preamble')
        self.add_line(f'\\addbibresource{{{self.bib_filename}.bib}}', 'preamble')
        self.add_line(f'\\title{{{self.title}}}', 'preamble')
        self.add_line('\\begin{document}', 'preamble')
        self.add_line('\date{}', 'preamble')
        self.add_line('\maketitle', 'preamble')
        if self.table_of_contents:
            self.add_line('\\tableofcontents', 'preamble')        
        self.add_line('\\printbibliography', 'endings')
        self.add_line('\\end{document}', 'endings')


def add_image_to_doc(
    fig: Union[MplFig, PlotlyFig], 
    filename: str, 
    filetype: str,
    title: str, 
    tex_doc: StandardTexDoc, 
    section: str,
    subsection: str='',
    caption: str='',
    fig_width: float=0.85,
):
    """
    Save a figure image to a local directory and include in TeX doc.

    Args:
        fig: The figure object
        filename: A string for the filename to save the figure as
        filetype: The extension to determine the type of figure
        caption: Figure caption for the document
        tex_doc: The working document
        section: Pass through to include_figure method to the document writing object
        subsection: Pass through
        caption: Pass through
        fig_width: Pass through

    Raises:
        TypeError: If the figure is not one of the two supported formats
    """
    if not hasattr(tex_doc, 'path'):
        return
    fig_folder = 'figures'
    fig_path = tex_doc.path / fig_folder
    fig_path.mkdir(exist_ok=True)
    full_filename = f'{filename}.{filetype}'
    if isinstance(fig, np.ndarray):
        plt.savefig(fig_path / full_filename)
    elif isinstance(fig, MplFig):
        fig.savefig(fig_path / full_filename)
    elif isinstance(fig, PlotlyFig):
        fig.write_image(fig_path / full_filename)
    else:
        raise TypeError('Figure type not supported')
    tex_doc.include_figure(title, filename, filetype, fig_folder, section, subsection=subsection, caption=caption, fig_width=fig_width)
