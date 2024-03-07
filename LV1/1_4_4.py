fhand = open("song.txt")
wordcounts = {}
for line in fhand :
    line = line.rstrip()
    # print (line)
    words = line.split()
    for word in words:
        # print(word)
        word = word.lower()
        if word in wordcounts.keys():
            wordcounts[word] += 1
        else:
            wordcounts[word] = 1
            
fhand.close()
print("Rijeci koje se jednom ponavljaju")
counter = 0;
for key in wordcounts.keys():
    if wordcounts[key] == 1:
        counter += 1
        print(key)
print(f"Broj rijeci koje se jednom ponavljaju: {counter}")

