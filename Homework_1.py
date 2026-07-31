def findmean(x):
    return sum(x)/len(x)

x =[2, 4, 6, 8]

y = [100, 120, 140, 160]

a = findmean(x)
b = findmean(y)

num = 0
deno = 0

for i in range(len(x)):
    num = num+(x[i]-a)*(y[i]-b)
    deno = deno+pow((x[i]-a), 2)
    slope = num/deno
    c = round(b-(slope*a), 2)
    a1 = 10
    prediction1 = slope*a1+c
    a2 = 20
    prediction2 = slope*a2+c
    a3 = 30
    prediction3 = slope*a3+c
    a4 = 40
    prediction4 = slope*a4+c
    a5 = 50
    prediction5 = slope*a5+c

print(prediction1)
print(prediction2)
print(prediction3)
print(prediction4)
print(prediction5)