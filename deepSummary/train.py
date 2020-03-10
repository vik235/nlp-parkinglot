import torch
import torch.nn as nn
import time

def train(model, iterator, optimizer, criterion, clip):
    
    print('*' * 40, ' Starting a training epoch ', '*' * 40)

    model.train()
    
    epoch_loss = 0
    time_started = time.time()
    for i, batch in enumerate(iterator):
        
        start_time = time.time()
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        start_mins, start_secs = epoch_time(time_started, end_time)
        epoch_loss += loss.item()
        if(i % 100 == 0):
            print(f'Batch: {i} | Time: {epoch_mins}m {epoch_secs}s ||  Time Since Start: {start_mins}m {start_secs}s')
        
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, context):
    
    print('*' * 40, context, 'evaluation', '*' * 40)
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):
            
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
            
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
