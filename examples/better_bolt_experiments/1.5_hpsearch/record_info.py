"""
Script that records scores, epochs, and number of tasks
"""


def record_info(file_name, train, valid, weight, combo):
  file_object = open(file_name, "a")
  for value in train[1].values():
    for score in value:
      file_object.write(str(score) + ',')

  file_object.write('train:,')
  for value in valid[1].values():
    for score in value:
      file_object.write(str(score) + ',')
  file_object.write(str(weight) + ',' + str(combo) + '\n')
  """
  file_object.write(
      str(next(iter(train[1].values()))) + "," +
      str(next(iter(valid[1].values()))) + ',' + str(epochs) + "\n")
  """
  print("done")
