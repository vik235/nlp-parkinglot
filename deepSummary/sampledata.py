import os 
import csv
import random

sample_size_train = 1937135
sample_size_valid = 10000
sample_size_test = 10000


functionList = list()
pidfunctions = list() # Keeps the project Id for tracking / error analysis
with open(file = './funcom_tokenized/train/functions.train', ) as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    for row in reader:
        functionList.append(row[1])
        pidfunctions.append(row[0])

commentList = list()
pidComments = list() # Keeps the project Id for tracking / error analysis 
with open(file = './funcom_tokenized/train/comments.train', ) as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    for row in reader:
        commentList.append(row[1])
        pidComments.append(row[0])

print(len(commentList))
print(len(functionList))

sample = random.sample(range(1, len(functionList)), sample_size_train)

train_idx = sample

funcList = [ functionList[i] for i in train_idx]
funcPidList = [ pidfunctions[i] for i in train_idx]

commList = [ commentList[i] for i in train_idx]
pidComments = [ pidComments[i] for i in train_idx]

with open(file = './data/functions.train.src', mode = 'w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([funcList])
with open(file = './data/functions.train.tgt', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([commList])
with open(file = './data/pids/functions.train.pid.src', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([funcPidList])
with open(file = './data/pids/functions.train.pid.tgt', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([pidComments])

#############################################################################################

functionList = list()
pidfunctions = list() # Keeps the project Id for tracking / error analysis
with open(file = './funcom_tokenized/valid/functions.valid', ) as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    for row in reader:
        functionList.append(row[1])
        pidfunctions.append(row[0])

commentList = list()
pidComments = list() # Keeps the project Id for tracking / error analysis 
with open(file = './funcom_tokenized/valid/comments.valid', ) as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    for row in reader:
        commentList.append(row[1])
        pidComments.append(row[0])

print(len(commentList))
print(len(functionList))

sample = random.sample(range(1, len(functionList)), sample_size_valid)

train_idx = sample

funcList = [ functionList[i] for i in train_idx]
funcPidList = [ pidfunctions[i] for i in train_idx]

commList = [ commentList[i] for i in train_idx]
pidComments = [ pidComments[i] for i in train_idx]

with open(file = './data/functions.valid.src', mode = 'w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([funcList])
with open(file = './data/functions.valid.tgt', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([commList])
with open(file = './data/pids/functions.valid.pid.src', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([funcPidList])
with open(file = './data/pids/functions.valid.pid.tgt', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([pidComments])


#####################################################################################################
functionList = list()
pidfunctions = list() # Keeps the project Id for tracking / error analysis
with open(file = './funcom_tokenized/test/functions.test', ) as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    for row in reader:
        functionList.append(row[1])
        pidfunctions.append(row[0])

commentList = list()
pidComments = list() # Keeps the project Id for tracking / error analysis 
with open(file = './funcom_tokenized/test/comments.test', ) as f:
    reader = csv.reader(f, dialect='excel', delimiter='\t')
    for row in reader:
        commentList.append(row[1])
        pidComments.append(row[0])

print(len(commentList))
print(len(functionList))

sample = random.sample(range(1, len(functionList)), sample_size_test)

train_idx = sample

funcList = [ functionList[i] for i in train_idx]
funcPidList = [ pidfunctions[i] for i in train_idx]

commList = [ commentList[i] for i in train_idx]
pidComments = [ pidComments[i] for i in train_idx]

with open(file = './data/functions.test.src', mode = 'w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([funcList])
with open(file = './data/functions.test.tgt', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([commList])
with open(file = './data/pids/functions.test.pid.src', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([funcPidList])
with open(file = './data/pids/functions.test.pid.tgt', mode ='w') as f:
    writer = csv.writer(f, delimiter='\n', quotechar='\"', quoting=csv.QUOTE_MINIMAL)
    writer.writerows([pidComments])



