
import os
import csv
import operator
import itertools


# Open all files in given directory, call a method to return edited version of these files
# Processes / edits the training files, trains out dictionary of cue words

def iterate_all_files(input_path, output_path):
    directory = os.path.join(input_path)
    
    cue_words = {}
    
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                edit_training_file(input_path + file, output_path + file)
                cue_words = baseline_cues(output_path + file, cue_words)
    return cue_words


def edit_training_file(input_path, output_path):
    with open(input_path,'rb') as tsvin, open(output_path, 'wb') as tsvout:
        tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
        tsvout = csv.writer(tsvout, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')

        last_row_end = '-'

        for row in tsvin:
            if row == []:
                tsvout.writerow('')
            else:    
                if row[2] == '_':
                    tag = 'O'
                else:
                    if last_row_end == row[2]:
                        tag = 'I'
                    else:
                        tag = 'B'
                tsvout.writerow([row[0],row[1],tag])
                last_row_end = row[2]


def baseline_cues(file_name, cue_words):
    with open(file_name) as tsv:
        for line in csv.reader(tsv, delimiter='\t', quoting=csv.QUOTE_NONE):
            if (line!=[] and  line[2] != 'O'):
                if (line[0] in cue_words):
                    cue_words[line[0]] = cue_words.get(line[0])+1
                else:
                    cue_words[line[0]] = 1
    return cue_words

# For each document in test, for each line, if line[0] in sorted_cues: attach a cue (logic for b vs i), else: == 'O'

def iterate_all_test_files(input_path, output_path, cue_dict):
    directory = os.path.join(input_path)
    
    counter = 0
    indices = []
    sent_counter = 0
    sent_indices = []

    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                [counter, indices, sent_counter, sent_indices] = edit_test_file(input_path + file, output_path + file, cue_dict,                                                     counter, indices, sent_counter, sent_indices)
    return counter, indices, sent_indices


def edit_test_file(input_path, output_path, cue_dict, counter, index_values, sentence_counter, sentence_indices):
    with open(input_path,'rb') as tsvin, open(output_path, 'wb') as tsvout:
        tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
        tsvout = csv.writer(tsvout, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')

        last_row_end = 'O'
        uncertain_count = 0 # does this sentence have an uncertain word in it

        for row in tsvin:
            if row != []:
                if row[0] in cue_dict:
                    uncertain_count += 1
                    index_values.append(counter)
                    if last_row_end == 'B':
                        tag = 'I'
                    else:
                        tag = 'B'
                else:
                    tag = 'O'
                tsvout.writerow([row[0],row[1],tag])
                last_row_end = tag
                counter += 1
            else:
                tsvout.writerow('')
                if uncertain_count > 3: # set this to the number of desired cue words to consider a sentence "uncertain"
                    sentence_indices.append(sentence_counter)
                uncertain_count = 0
                sentence_counter += 1
    return counter, index_values, sentence_counter, sentence_indices


'''
Training the model
'''

in_path = 'train/'
out_path = 'train-edited/' # must make the train-edited folder in the same directory


cues = iterate_all_files(in_path, out_path)
sorted_cues = dict(sorted(cues.iteritems(), key=lambda x:-x[1])[:200])
# print sorted_cues

'''
Editing the public test docs
'''

public_in_path = 'test-public/'
public_out_path = 'test-public-edited/' # must make this folder too


[public_counter, public_indices, public_sent_indices] = iterate_all_test_files(public_in_path, public_out_path, sorted_cues)


# print public_counter
# print len(public_indices)
# print len(public_sent_indices)


'''
Editing the private test docs
'''


private_in_path = 'test-private/'
private_out_path = 'test-private-edited/' # make this folder too
[private_counter, private_indices, private_sent_indices] = iterate_all_test_files(private_in_path, private_out_path, sorted_cues)


# print private_counter
# print len(private_indices)
# print len(private_sent_indices)


'''
Create the output file for Kaggle submission 1
'''

# create_ranges adapted from stackoverflow post for generating ranges:
# http://stackoverflow.com/questions/4628333/converting-a-list-of-integers-into-range-in-python
def create_ranges(i):
    for a, b in itertools.groupby(enumerate(i), lambda (x, y): y - x):
        b = list(b)
        yield b[0][1], b[-1][1]

def format_ranges(i):
    ranges = []
    for tup in i:
        ranges.append(str(tup[0]) + "-" + str(tup[1]))
    return ranges


'''
Create the csv file for sequence predictions
'''

public_range_list = list(create_ranges(public_indices))
private_range_list = list(create_ranges(private_indices))
public_formatted = " ".join(format_ranges(public_range_list))
private_formatted = " ".join(format_ranges(private_range_list))

with open('sequence_predictions.csv', 'wb') as output:
    output = csv.writer(output)
    output.writerow(['Type', 'Spans'])
    output.writerow(['CUE-public', public_formatted])
    output.writerow(['CUE-private', private_formatted])

'''
Create the csv file for sentence-level predictions
'''

with open('sentence_predictions.csv', 'wb') as output:
    output = csv.writer(output)
    output.writerow(['Type', 'Indices'])
    output.writerow(['SENTENCE-public', " ".join([str(i) for i in public_sent_indices])])
    output.writerow(['SENTENCE-private', " ".join([str(i) for i in private_sent_indices])])

