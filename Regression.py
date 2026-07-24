def findmean(x):
    return sum(x)/len(x)

#Number of study hours:
x =[1, 2, 3, 4, 5]
#Marks earned:
y = [60, 70, 80, 90, 100]

a = findmean(x)
b = findmean(y)

num = 0
deno = 0

for i in range(len(x)):
    num = num+(x[i]-a)*(y[i]-b)
    deno = deno+pow((x[i]-a), 2)
    slope = num/deno
    c = round(b-(slope*a), 2)
    hours = 20
    predicted_marks = slope*hours+c

print(predicted_marks)