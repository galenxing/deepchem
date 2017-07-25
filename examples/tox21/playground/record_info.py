"""
Script that records scores, epochs, and number of tasks
"""


def record_info(file_name, train, valid, test, epochs, num_tasks, tasks):
  file_object = open(file_name, "a")
  file_object.write(
      str(train.values()) + "," + str(valid.values()) + "," +
      str(test.values()) + "," + str(epochs) + "," + str(num_tasks) + "," +
      str(tasks) + "\n")
  print("done")
