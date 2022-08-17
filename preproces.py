import os
from collections import Counter
    
def main():
    result = []
    with open(os.path.join(os.getcwd(),'._1b_benchmark.dev.tokens'), encoding="UTF8") as f:
        file = f.read()
        sentences = file.split("\n")
        l = file.split()
        f.close()
    for sentence in sentences:
        result.append(sentence.split())

        #print(result)

    dic = Counter(l)

    for sentence in result:
        for i in range(len(sentence)):
            if dic[sentence[i]] < 3:
                sentence[i] = "<UNK>"

        #print(len(set()))

    with open(os.path.join(os.getcwd(), "dev_v3.txt"), "a", encoding='UTF8') as t:
        for sentence in result:
            for word in sentence:
                print(word, file=t, end=' ')
            print("",file=t)
        t.close()


if __name__ == "__main__":
        main()