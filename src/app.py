from utils import safe_write
from chains import (
    product_manager_chain,
    tech_lead_chain,
    file_structure_chain,
    file_path_chain,
    code_chain,
    missing_chain,
    new_classes_chain
)


request = """
Build a software for AutoML in Pytorch. 
The code should be able to search the best architecture and hyperparameters for a CNN network based on different metrics.
You can use all the necessary packages you need.
"""

design = product_manager_chain.run(request)
print(design)

class_structure = tech_lead_chain.run(design)
print(class_structure)

file_structure = file_structure_chain.run(class_structure)
print(file_structure)

files = file_path_chain.run(file_structure)
print(files)

files_list = files.split('\n')

missing = True
missing_dict = {
    file: True for file in files_list
}

code_dict = {}

while missing:

    missing = False
    new_classes_list = []

    for file in files_list:

        if not missing_dict[file]:
            safe_write(file, code_dict[file])
            continue

        code = code_chain.predict(
            class_structure=class_structure,
            structure=file_structure, 
            file=file,
        )

        code_dict[file] = code
        response = missing_chain.run(code=code)
        missing_dict[file] = '<TRUE>' in response
        missing = missing or missing_dict[file]

        if missing_dict[file]:
            new_classes = new_classes_chain.predict(
                class_structure=class_structure,
                code=code
            )
            new_classes_list.append(new_classes)

    class_structure += '\n\n' + "\n\n".join(new_classes_list)