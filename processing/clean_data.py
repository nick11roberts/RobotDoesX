filename = '../data/data.txt'

out = open('../data/data_clean.txt', 'w')

with open(filename) as f:
    content = f.readlines()
    for line in content:
        if line[4].isdigit():
            out.write(line[5:].replace("\n", " "))
        else:
            out.write(line[4:].replace("\n", " "))
        
