# Reference
This package references: [bugFinder](https://github.com/penguinsAreFunny/bugFinder)
# Anaconda: Create Environment
First you need to create an environment from the dependencies file.
You can name your <env> as you like, f.e. ML:

    conda create --name <env> --file ./requirements.txt

The best way might be to use VSCode with JupyterNotebook extension.
Run the cells from main.ipynb with the newly create environment. 

Otherwise: Activate your environment and run the scripts. F.e. like that

    conda activate <env>
    jupyter notebook

# Further dependencies

You will need Java, if you want to use h2o AutoML. Every other cell will run without Java installed.
Do not miss to create the conda environment to run the scripts. @see: above. 

    java - used jdk 11.0.2
    
# Used Version
    Anaconda3 2021.05 (Python 3.8.8 64-bit)
