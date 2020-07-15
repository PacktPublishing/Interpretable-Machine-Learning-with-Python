import subprocess
import sys

def runcmd(cmd, verbose=False):
    sproc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, )
    output = ''
    numlines = 0
    error = True
    while True:
        if error:
            line = sproc.stderr.readline().decode("utf-8")
            if line == '' and (sproc.poll() is None or sproc.poll() == 0):
                error = False
        if not error:
            line = sproc.stdout.readline().decode("utf-8")
        if line == '' and sproc.poll() is not None:
            break
        if verbose:
            sys.stdout.write(line)
            sys.stdout.flush()
        output = output + line
        numlines = numlines + 1
    return error, output.strip(), numlines