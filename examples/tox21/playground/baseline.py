import os

tasks = [
    'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
    'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
]
num_epochs = 20

for task in tasks:
  for x in range(0, 3):
    command = "python tox21_tensorgraph_graph_conv.py " + task + " " + str(
        num_epochs)
    print(command)
    os.system(command)
