from itertools import chain
from collections import Counter

def shared_dimensions_frequency(file_with_dimensions,num_dimensions_removed):
    with open(file_with_dimensions) as f:
            lines = f.readlines()
            concepts = {}
            for line in lines:
                partition = line.split('[')
                value = partition[0].split(',')
                concepts[value[1].strip()] = partition[1].strip()[:-1].split(', ')

    list_of_dimensions = []
    for i in concepts:
        list_of_dimensions.append(concepts[i])

    items = Counter(chain.from_iterable(list_of_dimensions))

    count_of_itens = []
    for k, v in items.items():
        count_of_itens.append((k, v)) 

    count_of_itens.sort(key=lambda a: a[1], reverse=True)
    for i in count_of_itens[:num_dimensions_removed]:
        print(f' {i[0]}', end=',')
    print('done')

if __name__ == '__main__':
     method = 'shared_dimensions_frequency'
     num_dimensions_removed = 135 # how much dimensions will be removed

     if method == 'shared_dimensions_frequency':
          file_with_dimensions = 'results/pt-theta-001to005/135 dims/results_theta_0-05.txt'
          shared_dimensions_frequency(file_with_dimensions,num_dimensions_removed)