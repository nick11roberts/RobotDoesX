from pyspark import SparkContext

sc = SparkContext("local", "Clean Data")

infile = 'data/data.txt'
outfile = open('data/data_clean.txt', 'w')

def clean(line):
    if len(line) >= 5:
        if line[4].isdigit():
            if len(line) >= 6:
                return line[5:].replace("\n", " ")
            else:
                return ''
        else:
            return line[4:].replace("\n", " ")
    else:
        return ''

with open(infile) as f:
    content = f.readlines()
    '''
    for line in content:

    '''

    text_file = sc.textFile(infile)
    outdata = text_file.map(clean).reduce(lambda a, b: ''.join([a, b]))

    outfile.write(outdata)
