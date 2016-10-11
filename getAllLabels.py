from sets import Set

labels = set()

for line in open('testing_data.txt', 'r'):
    labels.add(line.split(' ')[1])

print labels
