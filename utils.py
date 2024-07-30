from IPython.display import Image, display

def display_graph(graph):
    # Generate a Mermaid PNG image of the graph with X-ray mode enabled
    graph_image = graph.get_graph(xray=True).draw_mermaid_png()
    
    # Display the graph image
    display(Image(graph_image))