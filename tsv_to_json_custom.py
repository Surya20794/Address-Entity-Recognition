import json
import logging
import sys
def tsv_to_json_format(input_path,output_path,unknown_label):
    try:
        f=open(input_path,'r') # input file
        fp=open(output_path, 'w') # output file
        data_dict={}
        annotations =[]
        label_dict={}
        s=''
        start=0
        
        for line in f:
            if line[0:len(line)-1]!='.\tO':
                word,entity=line.split('\t')
                #print(word,entity)
                s+=word+" "
                #print(s)
                entity=entity[:len(entity)-1]
                if entity!=unknown_label:
                    if len(entity) != 1:
                        #print(len(entity),"Yes")
                        d={}
                        d['text']=word
                        #print(d['text'])
                        d['start']=start
                        d['end']=start+len(word)-1  
                        #print(d['start'],d['end'])
                        try:
                            label_dict[entity].append(d)
                            #print(label_dict)
                        except:
                            label_dict[entity]=[]
                            label_dict[entity].append(d) 
                            #print(label_dict)
                start+=len(word)+1
            #else:
                data_dict['content']=s
                s=''
                #print(s)
                label_list=[]
                for ents in list(label_dict.keys()):
                    for i in range(len(label_dict[ents])):
                        if(label_dict[ents][i]['text']!=''):
                            l=[ents,label_dict[ents][i]]
                            for j in range(i+1,len(label_dict[ents])): 
                                if(label_dict[ents][i]['text']==label_dict[ents][j]['text']):  
                                    di={}
                                    di['start']=label_dict[ents][j]['start']
                                    di['end']=label_dict[ents][j]['end']
                                    di['text']=label_dict[ents][i]['text']
                                    l.append(di)
                                    label_dict[ents][j]['text']=''
                            label_list.append(l) 
                            #print(label_list)
                            
                #for entities in label_list:
                for entities in label_list:
                    label={}
                    label['label']=[entities[0]]
                    #print(label)
                    label['points']=entities[1:]
                    annotations.append(label)
                    #print(label)
                data_dict['annotation']=annotations
                annotations=[]
                #print(data_dict)
                json.dump(data_dict, fp)
                fp.write('\n')
                data_dict={}
                start=0
                label_dict={}
    except Exception as e:
        logging.exception("Unable to process file" + "\n" + "error = " + str(e))
        return None

#tsv_to_json_format("Data/ner_corpus_260.tsv",'Data/ner_corpus_260.json','abc')
tsv_to_json_format("Data/data2.tsv",'Data/data2.json','abc')



