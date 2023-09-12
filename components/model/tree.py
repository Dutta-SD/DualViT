from xml.etree import ElementTree as ET
from xml.etree.ElementTree import Element
from typing import Self


class LabelHierarchyTree:
    """Useful functions for getting hierarchical information from XML files"""

    def __init__(self, xml_path: str) -> Self:
        self.root = ET.parse(xml_path).getroot()

    def format_list_elements(self, elements: list[Element], names=False):
        return [
            (e.tag, e.attrib.get("label", None)) if not names else e.tag
            for e in elements
        ]

    def _get_at_depth(self, element, depth):
        elements = []
        if depth == 0:
            elements.append(element)
        else:
            for child in element:
                elements.extend(self._get_at_depth(child, depth - 1))
        return elements

    def get_elements_at_depth(self, depth: int, names=True):
        elements = self._get_at_depth(self.root, depth)
        return self.format_list_elements(elements, names)

    def get_immediate_children(self, tag: str, names=False):
        imm_children = []
        matching_elements = self.findall_matching(tag)
        for e in matching_elements:
            imm_children.extend([child for child in e])
        return self.format_list_elements(imm_children, names)

    def findall_matching(self, tag: str) -> list[Element]:
        """Returns all the elements with the given tag, including the root"""
        if self.root.tag == tag:
            return [self.root]
        xpath = f".//{tag}"
        matching_elements = self.root.findall(xpath)
        return matching_elements

    def is_leaf(self, tag: str | Element):
        if isinstance(tag, Element):
            return len(tag) == 0
        # Inefficient
        all_children = self.get_immediate_children(tag)
        return len(all_children) == 0

    def getall_leaves(self, tag: str, names=True):
        """Gets all the leaves given a particular node"""
        # Assume only 1 element present for matching tag
        leaves = []
        e = self.findall_matching(tag)[0]
        for node in e.iter():
            if self.is_leaf(node):
                leaves.append(node)
        return self.format_list_elements(leaves, names)


if __name__ == "__main__":
    ht = LabelHierarchyTree("components/data/cifar10.xml")
    print(ht.get_elements_at_depth(1))
    print(ht.get_immediate_children("light"))
    print(ht.getall_leaves("heavy"))
    print(ht.getall_leaves("dog"))
    print(ht.getall_leaves("class"))
