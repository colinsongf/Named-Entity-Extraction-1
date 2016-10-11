from nltk.tag.stanford import StanfordPOSTagger

english_postagger = StanfordPOSTagger('models/english-bidirectional-distsim.tagger', 'stanford-postagger.jar')


training_data_out = open('testing_data.txt', 'a')

# {tag : list index}
pos_tags_dict = {}
words_label_dict = {}
sentences = []

def createPOStagDict():
    count = 0
    for tag in open('pos.txt', 'r'):
        pos_tags_dict[tag.rstrip('\n')] = count
        count += 1

def extractDocuments(t_data):
    training_data = open(t_data, 'r')
    sentence = ""

    for line in training_data:
        line_split = line.split("\t")
        # Populate words_label_dict
        if line_split[0] != '\n':
            words_label_dict[line_split[0]] = line_split[1].rstrip('\n')
            sentence += line_split[0] + ' '
        else:
            sentences.append(sentence.split(' '))
            sentence = ""

    if sentence != "":
        sentences.append(sentence.split(' '))

def tagSentences(sentences):
    return english_postagger.tag_sents(sentences)

def createFeatureSet(word_tuple):
    size = len(pos_tags_dict) + 1
    feature_array = [0]*size
    word = word_tuple[0]
    pos = word_tuple[1]

    if word[0].isupper():
        feature_array[size - 1] = 1
    feature_array[pos_tags_dict[pos]] = 1

    writeFeatureToFile(word, words_label_dict[word], feature_array)

def writeFeatureToFile(word, label, feature_array):
    training_data_out.write(word + ' ')
    training_data_out.write(label + ' ')
    for i in xrange(len(feature_array)):
        if i == len(feature_array) - 1:
            training_data_out.write(str(feature_array[i]) + '\n')
        else:
            training_data_out.write(str(feature_array[i]) + ',')



if __name__ == '__main__':
    createPOStagDict()
    extractDocuments('test_nwire.txt')
    tagged = tagSentences(sentences)

    print "DONE TAGGING"

    for sentences in tagged:
        for word_tuple in sentences:
            if word_tuple[0] != '':
                createFeatureSet(word_tuple)
