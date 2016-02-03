from nltk.corpus import treebank as wsj

#file_in=open('wsj_0003.pos','r')

#content=file_in.readlines()
#file_in.close()

#print content[:5]

file_out_tag=open('tagged_sent_sample','w')
file_out_sent=open('untagged_sent_sample','w')

out1=wsj.sents()[:20]
out2=wsj.tagged_sents()[:20]
line=''
for i in out1 :
	#file_out_sent.write('\n\n')
	line=' '.join(i)
	file_out_sent.write(line)
	file_out_sent.write('\n')
	#print line	

for i in out2 :
	file_out_tag.write('\n\n')
	words=''
	for j in i :
		#print j
		words='/'.join(j)
		#print words
		file_out_tag.write(words)
		file_out_tag.write('\n')
	
file_out_tag.close()
file_out_sent.close()

#print out2[len(out2)-1]




