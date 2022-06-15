import sys

class Log:
    def __init__(self, filename = 'outputs.txt', stream = sys.stdout, use_terminal = False):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, output):
        if use_terminal == True:
            self.terminal.write(output)
        else:
            pass
        
        self.log.write(output)

    def flush(self):
        pass
