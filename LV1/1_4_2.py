try:
    mark = float(input())
    if mark<0.0 or mark > 1.0:
        print("Wrong input")
    elif mark<0.6:
        print("F")
    elif mark<0.7:
        print("D")
    elif mark<0.8:
        print("C")
    elif mark<0.9:
        print("B")
    elif mark<1.0:
        print("A")

except:
    print("Wrong input")