from xml.etree import ElementTree as ET


class LabelHierarchyTree:
    def __init__(self, xml_path) -> None:
        self.root = ET.parse(xml_path).getroot()

    def format_list_elements(self, elements):
        return [(e.tag, e.attrib) for e in elements]

    def _get_at_depth(self, element, depth):
        elements = []
        if depth == 0:
            elements.append(element)
        else:
            for child in element:
                elements.extend(self._get_at_depth(child, depth - 1))
        return elements

    def get_elements_at_depth(self, depth: int):
        elements = self._get_at_depth(self.root, depth)
        return self.format_list_elements(elements)

    def get_immediate_children(self, element_name):
        new_name = f".//{element_name}"
        imm_children = []
        matching_elements = self.root.findall(new_name)
        for e in matching_elements:
            imm_children.extend([child for child in e])
        return self.format_list_elements(imm_children)


if __name__ == "__main__":
    ht = LabelHierarchyTree("components/data/cifar10.xml")
    print(ht.get_elements_at_depth(1))
    print(ht.get_immediate_children("light"))
