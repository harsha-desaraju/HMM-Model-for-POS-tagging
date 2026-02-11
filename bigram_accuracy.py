#Program to print accuracy for bigram tagger

def word_and_tag(phrase):
    for i in range(len(phrase)-1,0,-1):
        if phrase[i] == "/" :
            word = phrase[:i]
            tag = phrase[i+1:]
            break
    return [word,tag]

tag_list = []

with open("data/brown.txt", 'r') as file:
    for line in file:
        phrase_list = line.split()
        if phrase_list != []:
            for phrase in phrase_list:
                tag = word_and_tag(phrase)[1]

                if tag in tag_list:
                    pass
                else:
                    tag_list.append(tag)


                    
word_tag = {}

with open("bi_test_output.txt","r") as file:
    for line in file:
        word_list = line.split()
        word_tag[word_list[0]] = word_list[1]

tag_matrix = []

for i in range(len(tag_list)):
    tag_matrix.append([])
    for j in range(len(tag_list)):
        tag_matrix[i].append(0)

with open("data/brown-test.txt", 'r') as file:
    for line in file:
        phrase_list = line.split()
        if phrase_list != []:
            for phrase in phrase_list:
                word = word_and_tag(phrase)[0]
                tag = word_and_tag(phrase)[1]

                pre_tag = word_tag[word]

                if pre_tag in tag_list:
                    a = tag_list.index(tag)
                    b = tag_list.index(pre_tag)

                    temp = tag_matrix[b][a]
                    tag_matrix[b][a] = temp +1



#print(tag_matrix)

num = 0
deno = 0

for i in range(len(tag_matrix)):
    for j in range(len(tag_matrix[i])):
        deno = deno + tag_matrix[i][j]
        if i == j :
            num = num + tag_matrix[i][j]

print("The accuracy of the bigram tagger is",(num/deno)*100,"%")  
