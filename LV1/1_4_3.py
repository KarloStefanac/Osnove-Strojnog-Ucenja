arr = []
while(True):
    num = input()
    if num == 'Done':
        break
    if not num.isnumeric():
        print("Not a number")
    else:
        num = int(num)
        arr.append(num)
arr.sort()
print(f"Srednja vrijednost: {sum(arr)/len(arr)}")
print(f"Minimalna vrijednost: {min(arr)}")
print(f"Srednja vrijednost: {max(arr)}")
for el in arr:
    print(el)
