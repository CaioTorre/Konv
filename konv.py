import imageio 
import numpy as np
import sys
import argvParser as parser
import imageFetch as fetch
import math

def applyKernelConvolution(image, weights):
    #img_out = np.uint8(np.zeros([image.shape[0], image.shape[1]]))
    img_out = image.copy()
    wsum = weights[0][0] + weights[0][1] + weights[0][2] + weights[1][0] + weights[1][1] + weights[1][2] + weights[2][0] + weights[2][1] + weights[2][2]
    if wsum == 0: wsum = 1
    #print('Old weights='+repr(weights))
    for i in range(len(weights)):
        for j in range(len(weights[0])):
            weights[i][j] = weights[i][j] / wsum
    #print('New weights='+repr(weights))
    img_out =  weights[0][0] * image[0:-2,0:-2] + weights[1][0] * image[1:-1,0:-2] + weights[2][0] * image[2:,0:-2]
    img_out += weights[0][1] * image[0:-2,1:-1] + weights[1][1] * image[1:-1,1:-1] + weights[2][1] * image[2:,1:-1]
    img_out += weights[0][2] * image[0:-2,1:-1] + weights[1][2] * image[1:-1,1:-1] + weights[2][2] * image[2:,2:]
    #img_out = ((img_out // 2) + 127.5)
    
    #img_out -= np.min(img_out)
    #img_out = img_out * (np.max(img_out) / np.min(img_out))
    img_out = img_out / (np.max(img_out) - np.min(img_out)) * 255
    img_out = img_out - np.min(img_out)
    #print('Returning image range=['+str(np.min(img_out))+','+str(np.max(img_out))+']')
    return np.uint8(img_out.clip(min=0,max=255))
    
base_convs = {
    'blur':  [[1,1,1],[1,1,1],[1,1,1]],
    'gBlur': [[1,2,1],[2,4,2],[1,2,1]],
    'hEdge': [[-1,-2,-1],[0,0,0],[1,2,1]],
    'vEdge': [[-1,0,1],[-2,0,2],[-1,0,1]],
    'edge':  [[0,1,0],[1,-4,1],[0,1,0]]
}

saveGrayScale = parser.exists(sys.argv, '--saveGS')
opcode = parser.getNextValue(sys.argv, '-m', None)

if opcode == None:
    kv = parser.getNextValue(sys.argv, '-k', None)
    if kv == None: print('No mode specified (-k)'); exit()

    img_in = fetch.getImageFromArgs('-s')
    
    if len(img_in.shape) > 2:
        img_greyscale = np.uint8((img_in[:,:,0] // 3 + img_in[:,:,1] // 3 + img_in[:,:,2] // 3))
    else:
        img_greyscale = img_in
        
    if saveGrayScale:
        imageio.imwrite('gs.png', img_greyscale)
    img_out = applyKernelConvolution(img_greyscale, base_convs[kv])

    imageio.imwrite('output_'+kv+'.png', img_out)
else:
    if opcode == 'mean':
        img1 = fetch.getImageFromArgs('-s1')
        img2 = fetch.getImageFromArgs('-s2')
        imageio.imwrite('output_mean.png', (img1 + img2) // 2)
    if opcode == 'dEdge':
        hEdge = (np.float64(fetch.getImageFromArgs('-hs')) - 127)
        vEdge = (np.float64(fetch.getImageFromArgs('-vs')) - 127)

        mags = np.sqrt(np.float64(hEdge * hEdge + vEdge * vEdge)) #/np.sqrt(2)
        for i in range(2):
            mags = applyKernelConvolution(mags, base_convs['blur'])
        
        mags = (mags - np.min(mags)) / (np.max(mags) - np.min(mags))
        #print('Mags='+repr(mags))        
        #print('Mags range = ['+str(np.min(mags)) +','+str(np.max(mags))+']')
        imageio.imwrite('magnitude.png', np.uint8(mags.clip(0,1) * 255))
        
        #threshold = 0.145
        #magsMask = np.clip(mags - threshold, a_min=0, a_max=1) / (1 - threshold)
        #print('MagsMask range = ['+str(np.min(magsMask)) +','+str(np.max(magsMask))+']')
        #imageio.imwrite('magnitude_clipped.png', np.uint8(magsMask * 255))
        
        hues = np.arctan2(hEdge, vEdge) * 180 / math.pi + 180
        img_out = np.zeros([mags.shape[0], mags.shape[1], 3])
        #print('Creating image sized ' + repr(mags.shape))
        for i in range(mags.shape[1] - 1):
            for j in range(mags.shape[0] - 1):
                #print('@' + str(i) + ',' + str(j))
                x = 1 - np.abs((hues[j][i] / 30) % 2 - 1)
                m = mags[j][i]
                if hues[j][i] < 60:
                    img_out[j,i,0] = 1 * m
                    img_out[j,i,1] = x * m
                    img_out[j,i,2] = 0 * m
                    
                else: 
                    if hues[j][i] < 120:
                        img_out[j,i,0] = x * m
                        img_out[j,i,1] = 1 * m
                        img_out[j,i,2] = 0 * m

                    else: 
                        if hues[j][i] < 180:
                            img_out[j,i,0] = 0 * m
                            img_out[j,i,1] = 1 * m
                            img_out[j,i,2] = x * m
                            
                        else: 
                            if hues[j][i] < 240:
                                img_out[j,i,0] = 0 * m
                                img_out[j,i,1] = x * m
                                img_out[j,i,2] = 1 * m
                                
                            else: 
                                if hues[j][i] < 300:
                                    img_out[j,i,0] = x * m
                                    img_out[j,i,1] = 0 * m
                                    img_out[j,i,2] = 1 * m
                                    
                                else:
                                    img_out[j,i,0] = 1 * m
                                    img_out[j,i,1] = 0 * m
                                    img_out[j,i,2] = x * m
                   
        imageio.imwrite('colorized.png', np.uint8((img_out * 255).clip(min=0,max=255)))
        #print('Hues: ' + repr(hues))
        #print('Hues range = ['+str(np.min(hues)) +','+str(np.max(hues))+']')
        #print('Max=' + str(np.max(hues)))
        #print('Min=' + str(np.min(hues)))
        
        
