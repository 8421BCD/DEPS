f=open('test_mistake.txt','r')
mistake1=0
mistake2=0
correct1=0
correct2=0
for line in f:
	mis, pre = line.strip().split('\t')
	if mis == '1':
		mistake1 += 1
		if pre == '1':
			correct1 += 1
	if mis == '0':
		mistake2 += 1
		if pre == '1':
			correct2 += 1
print('mistake: '+str(float(correct1)/float(mistake1)))
print('other: '+str(float(correct2)/float(mistake2)))
print('all: '+str(float(correct2+correct1)/float(mistake2+mistake1)))