#Program to get 1st 2000 sentences.

with open('brown.txt','r') as file:
    with open("brown-train.txt",'a') as file1:
        with open("brown-test.txt",'a') as file2:
            j=0
            for line in file:
                if j <500:
                    file2.write(line)
                    word_list = line.split()
                    if './.' in word_list:
                        j=j+1

                else:
                    file1.write(line)
                    
