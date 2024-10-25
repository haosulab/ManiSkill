from copy import deepcopy

import numpy as np


def string_to_array(string):
    """
    Converts a array string in mujoco xml to np.array.

    Examples:
        "0 1 2" => [0, 1, 2]

    Args:
        string (str): String to convert to an array

    Returns:
        np.array: Numerical array equivalent of @string
    """
    return np.array(
        [float(x) if x != "None" else None for x in string.strip().split(" ")]
    )


def find_elements(root, tags, attribs=None, return_first=True):
    """
    Find all element(s) matching the requested @tag and @attributes. If @return_first is True, then will return the
    first element found matching the criteria specified. Otherwise, will return a list of elements that match the
    criteria.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through.
        tags (str or list of str or set): Tag(s) to search for in this ElementTree.
        attribs (None or dict of str): Element attribute(s) to check against for a filtered element. A match is
            considered found only if all attributes match. Each attribute key should have a corresponding value with
            which to compare against.
        return_first (bool): Whether to immediately return once the first matching element is found.

    Returns:
        None or ET.Element or list of ET.Element: Matching element(s) found. Returns None if there was no match.
    """
    # Initialize return value
    elements = None if return_first else []

    # Make sure tags is list
    tags = [tags] if type(tags) is str else tags

    # Check the current element for matching conditions
    if root.tag in tags:
        matching = True
        if attribs is not None:
            for k, v in attribs.items():
                if root.get(k) != v:
                    matching = False
                    break
        # If all criteria were matched, add this to the solution (or return immediately if specified)
        if matching:
            if return_first:
                return root
            else:
                elements.append(root)
    # Continue recursively searching through the element tree
    for r in root:
        if return_first:
            elements = find_elements(
                tags=tags, attribs=attribs, root=r, return_first=return_first
            )
            if elements is not None:
                return elements
        else:
            found_elements = find_elements(
                tags=tags, attribs=attribs, root=r, return_first=return_first
            )
            deepcopy(elements)
            if found_elements:
                elements += (
                    found_elements if type(found_elements) is list else [found_elements]
                )

    return elements if elements else None
