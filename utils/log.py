import sys

class Log:
    def __init__(self, filename = 'outputs.txt', stream = sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, output):
        self.terminal.write(output)
        self.log.write(output)

    def flush(self):
        pass
