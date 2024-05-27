from setuptools import find_packages,setup
from typing import List


HYPHON_E_DOT= "-e ."
def get_requirements(filepath:str)->List[str]:
    requirements=[]

    with open(filepath)as file_obj:# we are useing this function for read requirements file ##
        requirements = file_obj.readline()
        requirements= [i.replace("/n"," ")for i  in requirements]

        if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)


setup(
name="New_ML_project_pipline",
version='0.0.1',
description=" Machine learning pipline project",
author= "Ravindra Amrao",
author_email= "ravindraamrao1986@gmail.com",
packages= find_packages(),
install_requires=get_requirements("requirements.txt")
)
