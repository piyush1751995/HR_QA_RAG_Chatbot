import os
from langchain_community.document_loaders import PyPDFLoader


if __name__=="__main__":

    # Load data from pdf url
    data_load = PyPDFLoader("https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf")
    data_test = data_load.load_and_split()
    print("No. of pages :", len(data_test))
    print(data_test[3])