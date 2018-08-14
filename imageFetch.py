import imageio
import sys
import argvParser as parser

def getImageFromArgs(cmd):
    path = parser.getNextValue(sys.argv, cmd, None)
    if path == None: print('Error: No input detected (' + cmd + ')'); exit()
    try:
        img = imageio.imread(path)
    except FileNotFoundError: print('Error: Image not found at ' + path); exit()
    return img
