import os

def flagparser(cmd):
    flags = {}
    args = cmd.split()

    for i in range(len(args)):        
        if(args[i] == '--train'):
            if(i+1 < len(args) and args[i+1].isnumeric()):
                flags['train'] = int(args[i+1])
            else:
                flags['train'] = False

        elif(args[i] == '--folder'):
            if(i+1 < len(args) and os.path.exists(args[i+1])):
                flags['folder'] = args[i+1]
            else:
                flags['folder'] = False
    
    return flags