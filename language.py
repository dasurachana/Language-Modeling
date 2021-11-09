"""
Language Modeling Project
Name:
Roll No:
"""

from os import linesep
import language_tests as test
import random
import numpy
import matplotlib
project = "Language" # don't edit this

### WEEK 1 ###

'''
loadBook(filename)
#1 [Check6-1]
Parameters: str
Returns: 2D list of strs
'''
def loadBook(filename):
    f=open(filename)
    line=f.read().splitlines()
    #print(line)
    list1=[]
    for i in line:
        if len(i)>0:
            x=i.split()
            list1.append(x)
    #print(list1)ss
    return list1

'''
getCorpusLength(corpus)
#2 [Check6-1]
Parameters: 2D list of strs
Returns: int
'''
def getCorpusLength(corpus):
    # print(corpus)
    total_length = sum(len(row) for row in corpus)
    #print(total_length)
    return total_length


'''
buildVocabulary(corpus)
#3 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def buildVocabulary(corpus):
    result_List=[]
    for each in corpus:
        for i in each:
            #print(each)
            if i not in result_List:
                result_List.append(i)
                #print(result_List)
    return result_List

'''
countUnigrams(corpus)
#4 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countUnigrams(corpus):
    unigram_Dict={}
    for each in corpus:
        for i in each:
            if i not in unigram_Dict:
                unigram_Dict[i]=1
            else:
                unigram_Dict[i]+=1
    #print(unigram_Dict)
    return unigram_Dict

'''
getStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: list of strs
'''
def getStartWords(corpus):
    new_List=[]
    for each in corpus:
        #for i in each:
        if each[0] not in  new_List:
            new_List.append(each[0])
            #print(each[0])       
    #print(new_List)
    return new_List


'''
countStartWords(corpus)
#5 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to ints
'''
def countStartWords(corpus):
    count_SartDict={}
    for each in corpus:
        if each[0] not in count_SartDict.keys():
            count_SartDict[each[0]]=1
        else:
            count_SartDict[each[0]]+=1
    #print(count_SartDict)
    return count_SartDict

'''
countBigrams(corpus)
#6 [Check6-1]
Parameters: 2D list of strs
Returns: dict mapping strs to (dicts mapping strs to ints)
'''
def countBigrams(corpus):
    bigram_Counts={}
    for sentence in corpus:
        #range(len(each)-1)
        for i in range(len(sentence)-1):
            #print(i)
            if sentence[i] not in bigram_Counts.keys() :
                bigram_Counts[sentence[i]]={}
            if sentence[i+1] not in bigram_Counts[sentence[i]]:
                bigram_Counts[sentence[i]][sentence[i+1]]=1
            else:
                bigram_Counts[sentence[i]][sentence[i+1]]+=1
        
    return bigram_Counts

### WEEK 2 ###

'''
buildUniformProbs(unigrams)
#1 [Check6-2]
Parameters: list of strs
Returns: list of floats
'''
def buildUniformProbs(unigrams):
    new_List=[]
    for each in unigrams:
        #print(each)
        uniformProbs=1/len(unigrams)
        new_List.append(uniformProbs)
    #print(new_List)
    return new_List



'''
buildUnigramProbs(unigrams, unigramCounts, totalCount)
#2 [Check6-2]
Parameters: list of strs ; dict mapping strs to ints ; int
Returns: list of floats
'''
def buildUnigramProbs(unigrams, unigramCounts, totalCount):
    prob_Of_Each=[]
    totalCount1 =0
    for each in unigramCounts:
        totalCount1 =unigramCounts[each]/ totalCount
        prob_Of_Each.append(totalCount1)
    return prob_Of_Each


'''
buildBigramProbs(unigramCounts, bigramCounts)
#3 [Check6-2]
Parameters: dict mapping strs to ints ; dict mapping strs to (dicts mapping strs to ints)
Returns: dict mapping strs to (dicts mapping strs to (lists of values))
'''
def buildBigramProbs(unigramCounts, bigramCounts):
    new_Dict={}
    for prevWord in bigramCounts:
        #print(bigramCounts[prevWord])
        words_List=[]
        probs_List=[]
        for keys in bigramCounts[prevWord]:
            words_List.append(keys)
            probs_List.append(bigramCounts[prevWord][keys]/unigramCounts[prevWord])
            temp={}
            temp["words"]=words_List
            temp["probs"]=probs_List
            new_Dict[prevWord]=temp

    return new_Dict


'''
getTopWords(count, words, probs, ignoreList)
#4 [Check6-2]
Parameters: int ; list of strs ; list of floats ; list of strs
Returns: dict mapping strs to floats
'''
def getTopWords(count, words, probs, ignoreList):
    highest_Prob_Dict={}
    temp_Dict={}
    for each in range(len(words)):
        temp_Dict[words[each]]=probs[each]
    #print(temp_Dict)
    while len(highest_Prob_Dict)!=count:
        highest_Count=0
        for i in temp_Dict:
            if temp_Dict[i]>highest_Count and i not in ignoreList and i not in highest_Prob_Dict:
                new_Variable=i
                highest_Count=temp_Dict[i]
        highest_Prob_Dict[new_Variable]=highest_Count

    return highest_Prob_Dict


'''
generateTextFromUnigrams(count, words, probs)
#5 [Check6-2]
Parameters: int ; list of strs ; list of floats
Returns: str
'''
from random import choices
def generateTextFromUnigrams(count, words, probs):
    word_Str=" "
    choice=choices(words, weights=probs, k=count)
    #print(choice)
    for each in choice:
        word_Str+=" "+each
    return word_Str


'''
generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs)
#6 [Check6-2]
Parameters: int ; list of strs ; list of floats ; dict mapping strs to (dicts mapping strs to (lists of values))
Returns: str
'''
def generateTextFromBigrams(count, startWords, startWordProbs, bigramProbs):
    bigram_Str=" "
    value=0
    last_Word=""
    while value!=count:
        if bigram_Str==" " or last_Word==".":
            for i in choices(startWords,weights=startWordProbs):
                last_Word=i
                bigram_Str+=" "+i
                value+=1
        else:
            for j in choices(bigramProbs[last_Word]["words"], weights=bigramProbs[last_Word]["probs"]):
                last_Word=j
                bigram_Str+=" "+j
                value+=1
    return bigram_Str


### WEEK 3 ###

ignore = [ ",", ".", "?", "'", '"', "-", "!", ":", ";", "by", "around", "over",
           "a", "on", "be", "in", "the", "is", "on", "and", "to", "of", "it",
           "as", "an", "but", "at", "if", "so", "was", "were", "for", "this",
           "that", "onto", "from", "not", "into" ]

'''
graphTop50Words(corpus)
#3 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTop50Words(corpus):
    return


'''
graphTopStartWords(corpus)
#4 [Hw6]
Parameters: 2D list of strs
Returns: None
'''
def graphTopStartWords(corpus):
    return


'''
graphTopNextWords(corpus, word)
#5 [Hw6]
Parameters: 2D list of strs ; str
Returns: None
'''
def graphTopNextWords(corpus, word):
    return


'''
setupChartData(corpus1, corpus2, topWordCount)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int
Returns: dict mapping strs to (lists of values)
'''
def setupChartData(corpus1, corpus2, topWordCount):
    return


'''
graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; str ; 2D list of strs ; str ; int ; str
Returns: None
'''
def graphTopWordsSideBySide(corpus1, name1, corpus2, name2, numWords, title):
    return


'''
graphTopWordsInScatterplot(corpus1, corpus2, numWords, title)
#6 [Hw6]
Parameters: 2D list of strs ; 2D list of strs ; int ; str
Returns: None
'''
def graphTopWordsInScatterplot(corpus1, corpus2, numWords, title):
    return


### WEEK 3 PROVIDED CODE ###

"""
Expects a dictionary of words as keys with probabilities as values, and a title
Plots the words on the x axis, probabilities as the y axis and puts a title on top.
"""
def barPlot(dict, title):
    import matplotlib.pyplot as plt

    names = []
    values = []
    for k in dict:
        names.append(k)
        values.append(dict[k])

    plt.bar(names, values)

    plt.xticks(rotation='vertical')
    plt.title(title)

    plt.show()

"""
Expects 3 lists - one of x values, and two of values such that the index of a name
corresponds to a value at the same index in both lists. Category1 and Category2
are the labels for the different colors in the graph. For example, you may use
it to graph two categories of probabilities side by side to look at the differences.
"""
def sideBySideBarPlots(xValues, values1, values2, category1, category2, title):
    import matplotlib.pyplot as plt

    w = 0.35  # the width of the bars

    plt.bar(xValues, values1, width=-w, align='edge', label=category1)
    plt.bar(xValues, values2, width= w, align='edge', label=category2)

    plt.xticks(rotation="vertical")
    plt.legend()
    plt.title(title)

    plt.show()

"""
Expects two lists of probabilities and a list of labels (words) all the same length
and plots the probabilities of x and y, labels each point, and puts a title on top.
Note that this limits the graph to go from 0x0 to 0.02 x 0.02.
"""
def scatterPlot(xs, ys, labels, title):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    plt.scatter(xs, ys)

    # make labels for the points
    for i in range(len(labels)):
        plt.annotate(labels[i], # this is the text
                    (xs[i], ys[i]), # this is the point to label
                    textcoords="offset points", # how to position the text
                    xytext=(0, 10), # distance from text to points (x,y)
                    ha='center') # horizontal alignment can be left, right or center

    plt.title(title)
    plt.xlim(0, 0.02)
    plt.ylim(0, 0.02)

    # a bit of advanced code to draw a y=x line
    ax.plot([0, 1], [0, 1], color='black', transform=ax.transAxes)

    plt.show()


### RUN CODE ###

# This code runs the test cases to check your work
if __name__ == "__main__":
    # test.testLoadBook()
    # test.testGetCorpusLength()
    # test.testBuildVocabulary()
    # test.testCountUnigrams()
    # test.testGetStartWords()
    # test.testCountStartWords()
    # test.testCountBigrams()
    # test.testBuildUniformProbs()
    # test.testBuildUnigramProbs()
    # test.testBuildBigramProbs()
    print("\n" + "#"*15 + " WEEK 1 TESTS " +  "#" * 16 + "\n")
    test.week1Tests()
    print("\n" + "#"*15 + " WEEK 1 OUTPUT " + "#" * 15 + "\n")
    test.runWeek1()

    ## Uncomment these for Week 2 ##
    

    print("\n" + "#"*15 + " WEEK 2 TESTS " +  "#" * 16 + "\n")
    test.week2Tests()
    print("\n" + "#"*15 + " WEEK 2 OUTPUT " + "#" * 15 + "\n")
    test.runWeek2()

    ## Uncomment these for Week 3 ##
"""
    print("\n" + "#"*15 + " WEEK 3 OUTPUT " + "#" * 15 + "\n")
    test.runWeek3()
"""