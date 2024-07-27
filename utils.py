from IPython.display import Image, display

def display_graph(graph):
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))