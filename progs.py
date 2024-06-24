# Prog network data
progs = {
    "nodes": [
        {"name": "Cops", "group": 1},
        {"name": "The Day This Morning", "group": 1},
        {"name": "Love Rats", "group": 2},
        {"name": "Big Feet", "group": 2},
        {"name": "Another Reality Show", "group": 2},
        {"name": "Star Wars Makeovers", "group": 3},
        {"name": "Film Night", "group": 3},
        {"name": "The Evening News", "group": 1},
        {"name": "Bored", "group": 3},
    ],
    "links": [
        {"source": 0, "target": 1, "value": 20},
        {"source": 0, "target": 2, "value": 30},
        {"source": 1, "target": 4, "value": 22},
        {"source": 6, "target": 2, "value": 5},
        {"source": 1, "target": 7, "value": 5},
        {"source": 3, "target": 8, "value": 15},
        {"source": 5, "target": 8, "value": 15},
    ],
}


# Create a handler for our read (GET) people
def read_nodes():
    """
    This function responds to a request for /api/nodes
    with the prog node data

    :return:        node data
    """
    return progs["nodes"]


def read_links():
    """
    This function responds to a request for /api/links
    with the prog links data

    :return:        link data
    """
    return progs["links"]
