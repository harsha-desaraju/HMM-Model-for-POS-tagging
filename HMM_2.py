#Program for tagging using HMM model - with tarnsition as emmission probablility


                                        # Part-I (Training the Model)


def word_and_tag(phrase):                               # Function for getting word and tag
    for i in range(len(phrase)-1,0,-1):
        if phrase[i] == "/" :
            word = phrase[:i]
            tag = phrase[i+1:]
            break
    return [word,tag]
    
        
def update(list1,word,list2,tag):                      # Function for populating master_list and master_pos_list
    a = list1.index(word)
    if tag in list2[a]:
        temp = list2[a][tag]
        list2[a][tag] = temp +1
    else:
        list2[a][tag] = 1


def atoi(char):                                        #Function for assigning a number to a given character
    if char.isupper():
        return ord(char)-65
    elif char.islower():
        return ord(char)-71
    else:
        return 52

def prob(prev_tag,Dict):                               # Function for populating trans_prob(matrix) using tag_list and prev_tag_list
    sum = 0
    for i in Dict:
        sum = sum + Dict[i]
    if prev_tag in Dict:
        return Dict[prev_tag]/sum
    else:
        return 0
            

master_list = []
master_pos_list = []

tag_list = []
prev_tag_list = []


for i in range(53):
    master_list.append([])
    master_pos_list.append([])


 # This code block populates master_list and Master_pos_list


with open("data/brown-train.txt", 'r') as file:
    for line in file:
        phrase_list = line.split()
        if phrase_list != []:
            prev_tag = '<s>'
            for phrase in phrase_list:
                a = atoi(phrase[0])
                word = word_and_tag(phrase)[0]
                tag = word_and_tag(phrase)[1]

                if word in master_list[a]:
                    update(master_list[a],word,master_pos_list[a],tag)

                else:
                    master_list[a].append(word)
                    master_pos_list[a].append({})
                    update(master_list[a],word,master_pos_list[a],tag)


                if tag in tag_list:
                    update(tag_list,tag,prev_tag_list,prev_tag)

                else:
                    tag_list.append(tag)
                    prev_tag_list.append({})
                    update(tag_list,tag,prev_tag_list,prev_tag)

                prev_tag = tag
                    

# This part forms the transition probability matrix

trans_prob = []                                          

new_tag_list = ["<s>"] + tag_list

for i in range(len(new_tag_list)):              
    trans_prob.append([])

for i in range(len(new_tag_list)):
    trans_prob[0].append(0)
    
for i in range(1,len(new_tag_list)):
    for j in range(len(new_tag_list)):
        trans_prob[i].append(prob(new_tag_list[j],prev_tag_list[i-1]))


# Transforms frequency matrix into probability matrix(for emmision probability)

for i in range(len(master_list)):                           
    for j in range(len(master_list[i])):                    
        sum = 0
        for k in master_pos_list[i][j]:
            sum = sum + master_pos_list[i][j][k]

        for k in master_pos_list[i][j]:
            master_pos_list[i][j][k] = master_pos_list[i][j][k]/sum


# This creates the confusion matrix

conf_tags = ['nn','jj','pps','nns','vb','rb','rp','in','cc' ,'cs','vbd','vbn','vbg','np']
confusion_matrix = []
for i in range(len(tag_list)):
    confusion_matrix.append([])
    for j in range(len(tag_list)):
        confusion_matrix[i].append(0)




                                              # Part-II (Testing the model)




        
# For a given sentence of the test data, this gives the emmision probability matrix       

with open("data/brown-test.txt", 'r') as file:
    total = 0
    correct = 0
    for line in file:
        phrase_list = line.split()
        if phrase_list != []:
            emmi_prob = []                                  
            prev_tag = '<s>'                               
            for i in range(len(tag_list)):                
                emmi_prob.append([])

            for i in range(len(phrase_list)):
                a = atoi(phrase_list[i][0])
                word = word_and_tag(phrase_list[i])[0]
                tag = word_and_tag(phrase_list[i])[1]

                if word in master_list[a]:
                    ind = master_list[a].index(word)

                    for j in range(len(tag_list)):
                        if tag_list[j] in master_pos_list[a][ind]:
                            emmi_prob[j].append(master_pos_list[a][ind][tag_list[j]])

                        else:
                            emmi_prob[j].append(0)

                else:
                    for j in range(len(tag_list)):
                        emmi_prob[j].append(trans_prob[j+1][new_tag_list.index(prev_tag)])

                prev_tag = tag
                

# This trace matrix is to store the links ( to trace back and get the tags)
                
            trace_matrix = []

            for i in range(len(tag_list)):
                trace_matrix.append([])


            for j in range(len(tag_list)):
                emmi_prob[j][0] = emmi_prob[j][0] * trans_prob[j+1][0]
              

# This is viterbi algorithm                   
            
            for i in range(1,len(phrase_list)):
                for j in range(len(tag_list)):
                    temp_list = []
                    if emmi_prob[j][i] != 0:
                        for k in range(len(tag_list)):
                            temp_list.append(emmi_prob[k][i-1]*trans_prob[j+1][k+1])      # intentionally omitted the prob. of the
                                                                                          # current tag
                        max_ind = temp_list.index(max(temp_list))
                      
                        trace_matrix[j].append(max_ind)

                        emmi_prob[j][i] = emmi_prob[j][i] * temp_list[max_ind]

                    else:
                        trace_matrix[j].append(0)

# This part is to get the tags of the words from tarce matrix. 
        
            temp_list = []        
            for j in range(len(tag_list)):
                temp_list.append(emmi_prob[j][-1])

            ind = temp_list.index(max(temp_list))

            sent = []

# This is to calculate the accuracy of the the tagger.

            for i in range(len(phrase_list)-1,-1,-1):
                total = total + 1
                word = word_and_tag(phrase_list[i])[0]
                tag = word_and_tag(phrase_list[i])[1]

                if tag == tag_list[ind]:
                    correct = correct + 1

               
                temp = confusion_matrix[tag_list.index(tag)][ind]
                confusion_matrix[tag_list.index(tag)][ind] = temp + 1
                    
                sent.append([word,tag_list[ind]])
                if i!=0:
                    ind = trace_matrix[ind][i-1]

            with open("output_2.txt",'a') as file:
                string = str(sent)
                file.write(string)
                file.write("\n\n\n")

                

# This code is for transforming the entries of confusion matrix into probabilities.
                
print("UNKNOWN -  Transition as Emission \n")

for i in range(len(confusion_matrix)):
    sum = 0
    for j in range(len(confusion_matrix[i])):
        sum = sum + confusion_matrix[i][j]
        
    for j in range(len(confusion_matrix[i])):
        temp = confusion_matrix[i][j]

        if sum!=0:
            confusion_matrix[i][j] = (temp/sum)*100
    

# This code is for printing the accuracy and confusion matrix. 
            
print('\t',end = '')
for i in conf_tags:
    print(i,'\t', end = '')
print('\n')

for i in range(len(conf_tags)):
    print(conf_tags[i],'\t',end = '')
    for j in range(len(conf_tags)):
        print('%.2f' % confusion_matrix[tag_list.index(conf_tags[i])][tag_list.index(conf_tags[j])],'\t',end = '')
    print('\n')


print("The accuracy is ", (correct/total)*100,'%' )
