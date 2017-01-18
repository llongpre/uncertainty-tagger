
# coding: utf-8

# In[28]:

''' TO-DO for project 4:
        Change the emission probabilities,
        cluster words, then when one word in a cluster receives uncertain tag,
        add +1 to count of all words in that cluster,
        also remember to add count to corresponding tag so probabilities still sum to 1
'''

import baseline
import os
import csv
from collections import Counter
import numpy as np
import baseline as bl
from nltk import cluster


# In[29]:

def create_cluster_dicts(inpt):
    
    ''' Hi Liane, just a few notes
    
    word_dict is the dictionary of {word: bitstring} so that when we come across a word in training,
        we can find out what it's cluster bitstring is.
    Then, with the bitstring, we can use it as a key in cluster_dict to get a list of all other words in the cluster
    
    Then, with the list of other words, we also add a bigram of (tag, word) to our list in the next function
        that eventually gets turned in emission probabilities.
    This way, we only make code to add more bigrams to our list. We don't actually need to change anything else.
    I have more notes about this in the function below
    '''
    
    with open(inpt,'rb') as tsvin:
        tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
        
        word_dict = {} # {word: bit string for cluster, ...}
        cluster_dict = {} # {bit string: [word in cluster, word, word,...], ...}
        
        for row in tsvin:
            bitstring = row[0]
            word = row[1] 
            word_dict[word] = bitstring
            if cluster_dict.get(bitstring):
                cluster_dict[bitstring].append(word)
            else:
                cluster_dict[bitstring] = [word]
        
    return word_dict, cluster_dict


# In[126]:

def iterate_files(input_path, word_bits, clusters):
    
    ''' Iterate through all files in a given training path and construct two lists,
    one of the tags in order and another of (tag, word) bigrams
    
    Hi again, TO-DO:
        When we come across a word with a B or I tag, we find out what it's bitstring is.
        Then with the bitstring we find the list of other words in the same cluster.
        Then, with the list of cluster words, we add a bigram of (tag, word) for each word to the tag_words list
            --This is how we add a count for each word in the cluster without having to modify other code
        
        Let me know if you have any questions.  Hope you're having a good break!
    '''
    
    directory = os.path.join(input_path)
    # print directory
    
    corpus = []

    thirds = [] # the third element in each row of each doc
    thirds_count = []
    tag_words = []
    tag_threshold = 1 # use this and a np.random.uniform() comparison to omit certain fractions of documents from training
    validation_files = {}
    validation_threshold = 0.9 # use to extract 1-threshold documents for validation
    added_tags = [] # same as thirds but also includes added cluster words
    add_tag_threshold = 0.15
    
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".txt") and np.random.uniform() < validation_threshold:
                temp_thirds = []
                temp_tag_words = []
                
                with open(directory+file,'rb') as tsvin:
                    tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
                    
                    sentence_thirds = []
                    sentence_tag_words = []
                    
                    for row in tsvin:
                        
                        if row != []:
                            corpus.append(row[0])
                            temp_thirds.append(row[2])
                            sentence_thirds.append(row[2])
                            temp_tag_words.append((row[2],row[0]))
                            sentence_tag_words.append((row[2],row[0]))

                            #When we come across a word with a B or I tag, we find out what its bitstring is.
                            #Then with the bitstring we find the list of other words in the same cluster.
                            #Then, with the list of cluster words, we add a bigram of (tag, word) for each word to the tag_words list
                            if row[2] != 'O' and row[0] in word_bits and np.random.uniform < add_tag_threshold:
                                bit_string = word_bits[row[0]]
                                other_words = clusters[bit_string]

                                for word in other_words:
                                    temp_tag_words.append((row[2],word))
                                    added_tags.append(word)
                                    thirds_count.append(row[2])
                            thirds_count.append(row[2])

                        else:
                            if 'B' in set(sentence_thirds):
                                temp_thirds += (2 * sentence_thirds)
                                temp_tag_words += (2 * sentence_tag_words)
                            sentence_thirds = []
                            sentence_tag_words = []
                
                # always add if 'B' is in the temp_thirds list
                if 'B' in set(temp_thirds):
                    thirds += temp_thirds
                    tag_words += temp_tag_words
                elif np.random.uniform() < tag_threshold: # if no uncertain cues, include if below threshold
                    thirds += temp_thirds
                    tag_words += temp_tag_words
                    
            else:
                with open(directory+file,'rb') as tsvin:
                    tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
                    
                    temp_sentence = []
                    sentences = []
                    sent_tags = []
                    tags = []
                    
                    for row in tsvin: 
                        if row != []:
                            temp_sentence.append(row[0])
                            sent_tags.append(row[2])
                        else:
                            sentences.append(temp_sentence)
                            tags.append(sent_tags)
                            sent_tags = []
                            temp_sentence = []
                
                validation_files[file] = (sentences, tags)
                    
    return thirds, thirds_count, tag_words, added_tags, validation_files, corpus


# In[ ]:




# In[127]:

def find_bigrams(input_list):
    
    ''' Create a list of bigrams from an input list '''
    
    return zip(input_list, input_list[1:])

def bigram_probs(bigrams, prefixes):
    
    ''' Calculate the bigram probabilities for an input list of tuples and list of the corresponding prefixes '''
    
    prefix_counts = Counter(prefixes)

    bigram_probs = {}
    bigram_counts = Counter(bigrams)
    
    for bigram in bigram_counts:
#         bigram_probs[bigram] = (bigram_counts[bigram] + 1) / (float(prefix_counts[bigram[0]]) + len(bigram_counts))
        bigram_probs[bigram] = (bigram_counts[bigram]) / (float(prefix_counts[bigram[0]]))
    return bigram_probs


# In[128]:

# For Viterbi, will take in list of strings (a sentence in a document), it will take in transition and emission probs
# output a list of tags that is as long as the input sequence

def viterbi(t_probs,e_probs,i_probs,states,sentence, train_length):
    state_indices = range(len(states))

    #initialize probability for first observation:
    item_probs = []
    for state in states:
        try:
            item_probs.append(i_probs[state]*e_probs[state,sentence[0]])
        except KeyError:
            item_probs.append(i_probs[state] * 1 / float(train_length))

    path = states   # initialize paths to state names

    for word in sentence[1:]:
        next_item_probs = []
        for i in state_indices:
            temp = [item_probs[i]*t_probs[states[i],states[j]] for j in state_indices]
            next_item_probs.append(temp)
            
        new_path=[]
        for i in state_indices:
            probs_into = [next_item_probs[j][i] for j in state_indices]
            try:
                item_probs[i] = max(probs_into)*e_probs[(states[i],word)]
            except KeyError:
                item_probs[i] = max(probs_into)*(1 / float(train_length))
                
            new_path.append(path[np.argmax(probs_into)] + states[i])
        path = new_path
    
    return path[np.argmax(item_probs)], max(item_probs)


# In[129]:

# To edit test documents:
# Need list of words (first item in each row), including where spaces are so that can write corresponding output

# function to make list of sequences from document (read from test-public, test-private folders)
# function to write these lists to the edited version of an input document (write to test-public-edited, etc.)

def get_sequences(input_path, transition, emission, initial_p, states, length):
    with open(input_path, 'rb') as test:
        test = csv.reader(test, delimiter='\t', quoting=csv.QUOTE_NONE)
        all_words = []
    
        for row in test:
            if row == []:
                all_words.append("")
            else:
                all_words.append(row[0])
        
        sentences = []
        temp = []
        i = 0
        for i in range(len(all_words)):
            if all_words[i] != '':
                temp.append(all_words[i])
            else:
                sentences.append(temp)
                temp = []
        
        sequences = []
        for sent in sentences:
            sequences.append(list(viterbi(transition, emission, initial_p, states, sent, length)[0]))
            
    return sequences


# In[130]:

def write_sequences(sequences, input_path, output_path):
    with open(input_path,'rb') as tsvin, open(output_path, 'wb') as tsvout:
        tsvin = csv.reader(tsvin, delimiter='\t', quoting=csv.QUOTE_NONE)
        tsvout = csv.writer(tsvout, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar='')
        
        sentence_counter = 0
        word_counter = 0
        
        for row in tsvin:
            if row == []:
                tsvout.writerow("")
                sentence_counter += 1
                word_counter = 0
            else:
                tsvout.writerow([row[0], row[1], sequences[sentence_counter][word_counter]])
                word_counter += 1


# In[131]:

def return_indices(file_path):
    directory = os.path.join(file_path)
    
    sentence_counter = 0
    index_counter = 0
    sentence_indices = []
    word_indices = []
    uncertain_count = 0
    
    for root,dirs,files in os.walk(directory):
        for file in files:
            if file.endswith(".txt"):
                with open(file_path+file, 'rb') as doc:
                    doc = csv.reader(doc, delimiter='\t', quoting=csv.QUOTE_NONE)
                    
                    for row in doc:
                        if row != []:
                            if row[2] != 'O':
                                word_indices.append(index_counter)
                                uncertain_count += 1
                            index_counter += 1
                        else:
                            if uncertain_count > 0:
                                sentence_indices.append(sentence_counter)
                            uncertain_count = 0
                            sentence_counter += 1
    return word_indices, sentence_indices


# In[ ]:


    


# In[132]:

''' Create cluster dicts '''

# TO-DO: we should compare cluster dicts from different files

cluster_file = 'paths_1000_min50.txt'
word_bit, clusters = create_cluster_dicts(cluster_file)
# clusters


# In[134]:

''' Train and validate model over multiple trials '''

trials = 10
total_tp = 0.
total_fp = 0.
total_fn = 0.

for i in range(trials):
    print "On trial %s..." % (i + 1)
    
    ''' Train the model '''
    
    # Arguments to pass into viterbi

    [tags, tag_counts, tag_words, add_tags, val_set, corpus] = iterate_files('train-edited/', word_bit, clusters)
    tag_bigrams = find_bigrams(tags)
    trained_len = len(tag_words)

    # transition probabilities
    T = bigram_probs(tag_bigrams, tags)
    T[('O', 'I')] = 1 - T[('O', 'O')] - T[('O', 'B')]

    # dictionary of P(word|tag)
    E = bigram_probs(tag_words, tag_counts)

    # Initial probabilities
    bio_counts = Counter(tags)
    bio_total = sum(bio_counts.values())
    initial = {}
    for item in bio_counts:
        initial[item] = bio_counts[item] / float(bio_total)


    state_names=['B','I','O']
    viterbi(T,E,initial,state_names,['Parviz','Yahaghi',"'s",'most','widely','distributed','recordings', 'outside','Iran','is'], trained_len)



    ''' 
        Validate model

            Segment the training documents, only train on N portion of them
                -this is just our existing procedure, but on a smaller number of documents
            Then, run our algorithm on the remaining 1-N portion of documents
                Preserve the original training documents for comparison,
                but also generate our own model-edited version of the document
            Then, compare our model-generated file to the original training document to see score:
                (P,R,F=(2PR)/(P+R)) or just %correct?


            When we go to generate our training lists, randomly pull out a number of documents
                to serve as the validation set.
            Record which documents we reserve for validation, record the sequence of tags from that document
            Run Viterbi on the documents, compare the output to the output of the step above
            Calculate performance metrics


            For each document in validation, record the list of tags from it,
            run Viterbi on each sentence, generate the HMM predictions for each document,
            compare this to the actual tags to calculate metrics
    '''

    tags_correct = 0.
    given_tags = ""
    generated_tags = ""

    for doc in val_set: # for each file in the validation set
        sentences = val_set[doc][0]
        actual_tags = val_set[doc][1]

        for i in range(len(sentences)): # for each sentence, run viterbi on sentence and compare output to actual tags
            sent_words = sentences[i]
            sent_tags = "".join(actual_tags[i])
            [sequence, prob] = viterbi(T,E,initial,state_names,sent_words, trained_len)
            given_tags += sent_tags
            generated_tags += sequence

    tp = 0.
    fp = 0.
    fn = 0.

    for i in range(len(given_tags)):
        if given_tags[i] == generated_tags[i]:
            tags_correct += 1
            if generated_tags[i] != 'O': # tp if tag != 'O'
                tp += 1
        else:
            if generated_tags[i] != 'O': # fp if generated != O when correct does
                fp += 1
            else:     # fn if generated tag = 'O' when real tag does not
                fn += 1

    total_tp += tp
    total_fp += fp
    total_fn += fn
    
    P = tp / (tp + fp)
    R = tp / (tp + fn)
    F = (2 * P * R) / (P + R)

#     print "True positives:", tp
#     print "False positives:", fp
#     print "False negatives:", fn

#     print "Percentage of correct tags:", tags_correct / len(given_tags)
#     print "Precision:", P
#     print "Recall:", R
#     print "F-score:", F

# average scores over all trials
print "\nCalculated over %s trials ---------" % trials

print "Average true positives:", total_tp / trials
print "Average false positives:", total_fp / trials
print "Average false negatives:", total_fn / trials

P = tp / (tp + fp)
R = tp / (tp + fn)
F = (2 * P * R) / (P + R)

print "Percentage of correct tags:", tags_correct / len(given_tags)
print "Precision:", P
print "Recall:", R
print "F-score:", F


# In[ ]:




# In[ ]:




# In[45]:

public_input = 'test-public/'
public_output = 'test-public-edited/'
private_input = 'test-private/'
private_output = 'test-private-edited/'
    
directory = os.path.join(public_input)

for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            public_sequences = get_sequences(public_input+file,T,E,initial,state_names,trained_len)
            write_sequences(public_sequences, public_input+file, public_output+file)

directory = os.path.join(private_input)            

for root,dirs,files in os.walk(directory):
    for file in files:
        if file.endswith(".txt"):
            private_sequences = get_sequences(private_input+file,T,E,initial,state_names,trained_len)
            write_sequences(private_sequences, private_input+file, private_output+file)


# In[ ]:




# In[46]:

public_indices, public_sent_indices = return_indices(public_output)
private_indices, private_sent_indices = return_indices(private_output)


# In[47]:

public_range_list = list(bl.create_ranges(public_indices))
private_range_list = list(bl.create_ranges(private_indices))
public_formatted = " ".join(bl.format_ranges(public_range_list))
private_formatted = " ".join(bl.format_ranges(private_range_list))


# In[48]:

with open('sequence_predictions.csv', 'wb') as output:
    output = csv.writer(output)
    output.writerow(['Type', 'Spans'])
    output.writerow(['CUE-public', public_formatted])
    output.writerow(['CUE-private', private_formatted])


# In[49]:

with open('sentence_predictions.csv', 'wb') as output:
    output = csv.writer(output)
    output.writerow(['Type', 'Indices'])
    output.writerow(['SENTENCE-public', " ".join([str(i) for i in public_sent_indices])])
    output.writerow(['SENTENCE-private', " ".join([str(i) for i in private_sent_indices])])


# In[ ]:




# In[ ]:




# In[ ]:



