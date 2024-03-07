fhand = open ('SMSSpamCollection.txt')
ham_count = 0
ham_words = 0
spam_count = 0
spam_words = 0
exclamation_count = 1
for line in fhand :
    line = line.rstrip()
    # print (line)
    words = line.split()
    if words[0] == 'ham':
        ham_count += 1
        ham_words += (len(words) - len(words[0]))
    elif words[0] == 'spam':
        spam_count += 1
        spam_words += (len(words) - len(words[0]))
        letters = [*words[len(words)-1]]
        if letters[len(letters)-1] == '!':
            exclamation_count +=1
fhand.close()

print(f"Prosjecan broj rijeci u ham porukama: {ham_words/ham_count}")
print(f"Prosjecan broj rijeci u spam porukama: {spam_words/spam_count}")
print(f"Broj spam poruka koje zavrsavaju sa !: {exclamation_count}")
