"""
Script that combines two datasets
"""


def combine(file1, file2):
  #open the first file and store in a file obj
  input1 = open(file1, "r")
  input2 = open(file2, "r")

  combo_name = file1 + file2
  combination = open(combo_name, "a")
