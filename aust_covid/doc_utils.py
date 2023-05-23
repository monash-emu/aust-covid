from pathlib import Path
import pylatex as pl
from pylatex import LineBreak
from pylatex.section import Section
from pylatex.utils import NoEscape
import matplotlib.figure as mpl
import plotly.graph_objects as go

BASE_PATH = Path(__file__).parent.parent.resolve()
SUPPLEMENT_PATH = BASE_PATH / "supplement"


def escape_refs(
    text: str,
) -> str:
    """
    Don't escape characters if they are needed for citations.

    Args:
        text: Text for document

    Returns:
        Revised text string
    """
    return NoEscape(text) if "\cite{" in text else text


class DocElement:
    """
    Abstract class for creating a model with accompanying TeX documentation.
    """
    def __init__():
        pass

    def emit_latex():
        pass


class TextElement(DocElement):
    """
    Write text input to TeX document using PyLaTeX commands.
    """
    def __init__(
            self, 
            text: str,
        ):
        """
        Set up object with text input.

        Args:
            text: The text to write
        """
        self.text = escape_refs(text)

    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Write the text to the document.

        Args:
            doc: The PyLaTeX object to add to
        """
        doc.append(self.text)


class FigElement(DocElement):
    """
    Add a figure to a TeX document using PyLaTeX commands.
    """
    def __init__(
            self, 
            fig_name: str,
            caption: str="",
            resolution: str="350px",
        ):
        """
        Set up object with figure input and other requests.

        Args:
            fig_name: The name of the figure to write
            caption: Figure caption
            resolution: Resolution to write to
        """
        self.fig_name = fig_name
        self.caption = caption
        self.resolution = resolution
    
    def emit_latex(
            self, 
            doc: pl.document.Document,
        ):
        """
        Write the figure to the document.

        Args:
            doc: The PyLaTeX object to add to
        """
        with doc.create(pl.Figure()) as plot:
            plot.add_image(self.fig_name, width=self.resolution)
            plot.add_caption(self.caption)


class TableElement(DocElement):
    
    def __init__(self, col_widths, input_table):
        self.col_widths = col_widths
        self.table = input_table

    def emit_latex(self, doc):
        with doc.create(pl.Tabular(self.col_widths)) as output_table:
            headers = [""] + list(self.table.columns)
            output_table.add_row(headers)
            output_table.add_hline()
            for index in self.table.index:
                content = [index] + [escape_refs(str(element)) for element in self.table.loc[index]]
                output_table.add_row(content)
                output_table.add_hline()
        doc.append(LineBreak())


def add_element_to_document(
    section_name: str, 
    element: DocElement, 
    doc_sections: dict,
):
    """
    Add a document element to the working document compilation object.

    Args:
        section_name: Name of the document section to add the element to
        element: The element to add
        doc_sections: The document to be added to
    """
    if section_name not in doc_sections:
        doc_sections[section_name] = []
    doc_sections[section_name].append(element)


def save_pyplot_add_to_doc(
    plot: mpl.Figure, 
    plot_name: str, 
    section_name: str, 
    working_doc: dict, 
    caption: str="", 
    dpi: float=250,
):
    """
    Save a matplotlib figure to a standard location and add it to the working document.
    
    Args:
        plot: The figure object
        plot_name: Name to assign the file
        section_name: Section to add the figure to
        working_doc: Working document
        caption: Optional caption to add
        dpi: Resolution to save the figure at
    """
    plot.savefig(SUPPLEMENT_PATH / f"{plot_name}.jpg", dpi=dpi)
    add_element_to_document(section_name, FigElement(plot_name, caption=caption), working_doc)


def save_plotly_add_to_doc(
    plot: go.Figure, 
    plot_name: str, 
    section_name: str, 
    working_doc: dict, 
    caption="", 
    scale=4.0,
):
    """
    Save a plotly figure to a standard location and add it to the working document.

    Args:
        plot: The figure object
        plot_name: Name to assign the file
        section_name: Section to add the figure to
        working_doc: Working document
        caption: Optional caption to add
        dpi: Resolution adjuster for saving the figure
    """
    plot.write_image(SUPPLEMENT_PATH / f"{plot_name}.jpg", scale=scale)
    add_element_to_document(section_name, FigElement(plot_name, caption=caption), working_doc)


def compile_doc(
    doc_sections: dict, 
    doc: pl.document.Document,
):
    """
    Compile the full PyLaTeX document from the dictionary
    of elements by section requested.

    Args:
        doc_sections: Working compilation of instruction elements
        doc: The TeX file to create from the elements
    """
    for section in doc_sections:
        with doc.create(Section(section)):
            for element in doc_sections[section]:
                element.emit_latex(doc)
