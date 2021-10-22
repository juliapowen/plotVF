import numpy as np
import pickle
import scipy 
from scipy import signal
import cv2
import matplotlib.pyplot as plt
import json

def ensureMatrix(DLS, eye):
    #If DLS is matrix: Do nothing 
    flag = False
    print(DLS.shape)
    if len(DLS.shape) == 2:
        
        if DLS.shape[0] == 1: 
            print("Loading Vector")
            flag = True
         
        if DLS.shape == (8, 10):
            flag = True #{24-2 format}

        elif DLS.shape == (10, 10):
            flag = True #{30-2 format}

#         else:
#             print("Error 1: Matrix dimensions not recognized")
        if flag == False: 
            print("Error 1: Matrix dimensions not recognized")
            return
        DLSgrid = DLS
        return DLSgrid
    
    else:
    ########## Checking DLS.size (elements in vector)##########
        if DLS.size == 54: #24-2
            #Eye parameter should be = 'left', 'right', or 'ivf'
            if eye == 'left':            
                #slow way to create a mask:
                #I tried to copy the shape directly but Nan values aren't recognized. 
            
                mask = np.ones((8,10))*-1
                mask[:, 3:7] = 1
                mask[1:7, 2] = 1
                mask[1:7, 7] = 1
                mask[2:6, 1] = 1
                mask[2:6, 8] = 1
                mask[3:5, 9] = 1
            
            elif eye == 'right':
                mask = np.ones((8,10))*-1
                mask[:, 3:7] = 1
                mask[1:7, 2] = 1
                mask[1:7, 7] = 1
                mask[2:6, 1] = 1
                mask[3:5, 0] = 1
                mask[2:6, 8] = 1
            
            elif eye == 'ivf':
                mask = np.ones((8,10))*-1
                mask[:, 3:7] = 1
                mask[1:7, 2] = 1
                mask[1:7, 7] = 1
                mask[2:6, 1] = 1
                mask[3:5, 0] = 1
                mask[2:6, 8] = 1
                mask[3:5, 9] = 1
            
            else: 
                print(f'ERROR 2: Eye parameter not recognized: {eye}')
                return
            
            
        elif DLS.size == 76: #30-2
            mask = np.ones((10,10))*-1
            mask[:, 3:7] = 1
            mask[1:9, 2] = 1
            mask[1:9, 7] = 1
            mask[2:8, 1] = 1
            mask[2:8, 8] = 1
            mask[3:7, 0] = 1
            mask[3:7, 9] = 1
    
        elif DLS.size == 68: #10-2
            mask = np.ones((10,10))*-1
            mask[:, 4:6] = 1
            mask[1:9, 2:4] = 1
            mask[1:9, 6:8] = 1
            mask[2:8, 1] = 1
            mask[2:8, 8] = 1
            mask[4:6, 0] = 1
            mask[4:6, 9] = 1        
    
        else:
            print("Error 3: vector format not recognized")
            return
        

##########   VALIDATE VECTOR SIZES   ##########
    #np.where(mask>0)
    #if DLS.size != int(np.nansum(mask)):
    if DLS.size != len(sum(np.where(mask>0))):
        print("FAILURE: Number of elements in DLS Matrix and Mask are Mismatched")
        return
         
    else:
        print("SUCCESS, # Elements in DLS Matrix Matches the Mask")
    
    
##########   INSERT VALUES   ##########    
    #print("MASK", mask)
    mask[np.where(mask > 0)] = DLS

    DLSgrid = np.matrix(mask)
    print(DLSgrid.shape)
    return DLSgrid

def computeIVF(DLS_Left, DLS_Right, method):
    # Combine (either Mean or Max)
    ######do we need to flip left?######
    if method == 'mean':
        DLS_IVF = np.mean(np.stack((np.fliplr(DLS_Left),DLS_Right)),axis=0)
        return;
    elif method == 'max':
        DLS_IVF = np.max(np.stack((np.fliplr(DLS_Left),DLS_Right)),axis=0)
    else:
        error('Ungrecognised method: %s', method);
        return
    return DLS_IVF
            

def replacemissingvalues(DLSgrid,neighbours):
    while len(np.where(DLSgrid == -1.)[0]) > 0:
        
        #create matrix where all non zeros VALUES are equal to 1 
        hvf_values = np.zeros((np.shape(DLSgrid)))
        hvf_values[np.where(DLSgrid >=0)] = 1

        #create matrix where all outer (-1s) are equal to 1 
        nan_values = np.zeros(np.shape(DLSgrid))
        #nan_values[np.where(DLS_Left >=0)] = 0
        nan_values[np.where(DLSgrid < 0)] = 1

        #convolve HVF_values using neighbours kernel
        y = scipy.signal.convolve2d(hvf_values, neighbours, mode='same') * nan_values


        #change -1 to 0
        DLSgrid_zeros = np.copy(DLSgrid)
        DLSgrid_zeros[np.where(DLSgrid < 0)] = 0

        #6 convolve DLSgrid and replace only the max values
        out = scipy.signal.convolve2d(DLSgrid_zeros, neighbours, mode='same')/ y
        DLSgrid[np.where(y==np.max(y))] = out[np.where(y==np.max(y))]
        
    return DLSgrid

def getmask(format_dls, eye):
    if format_dls == '24':
        mask = np.array([[0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0]])
        
        # tweak mask if looking at left eye or IVF
        if eye == 'left':
            mask = np.fliplr(mask);
        elif eye == 'ivf':
            mask = mask + np.fliplr(mask);
            mask[np.where(mask==2)] = 1        

    elif format_dls == '30':
        mask = np.array([[0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
        [0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        ])

    return mask

def plotVF(DLS_Left, DLS_Right, eye, savestring):

    neighbours = np.array([[0,1,0],[1,0,1],[0,1,0]])
    
    if eye == 'left':
        DLS = DLS_Left
        DLS = DLS.astype('float64')
        DLSgrid = ensureMatrix(DLS, eye)
    elif eye == 'right':
        DLS = DLS_Right
        DLS = DLS.astype('float64')
        DLSgrid = ensureMatrix(DLS, eye)
    elif eye == 'ivf':
        DLS_Left = np.array(DLS_Left)
        DLS_Right = np.array(DLS_Right)
        DLS_Left = DLS_Left.astype('float64')
        DLS_Right = DLS_Right.astype('float64')

        DLS_Leftgrid = ensureMatrix(DLS_Left, eye)
        DLS_Leftgrid = ensureMatrix(DLS_Right, eye)

        ivf_method = 'max'
        DLSgrid = computeIVF(DLS_Left, DLS_Right, ivf_method)
    else:
        print('Error: unknown eye')
    
   
    f = open('tiles.json',)
    data = json.load(f)
    tiles = {k:np.array(v) for k,v in data.items()}

    
    DLSgrid_replaced = replacemissingvalues(DLSgrid,neighbours)

    #def placetiles(DLSgrid, tiles, eye):
    # detect format (24-2 or 30-2) --------------------------------
    if np.shape(DLSgrid_replaced) == (8,10):
        format_dls = '24';
    elif np.shape(DLSgrid_replaced) == (10,10):
        format_dls = '30';
    else:
        print('unknown format');
        return

    #get the mask based on the dimensions of the DLS adn eye
    mask = getmask(format_dls, eye)
    dim = (np.shape(mask)[1],np.shape(mask)[0])
    #upsample DLS grid to size of the mask using linear interpolation
    Vq = cv2.resize(DLSgrid_replaced.astype('float64'), dim)

    #place tiles to make image
    xf = 28; yf = 24;
    #initalize img to zeros shape (xf*np.shape(Vq)[0], yf*np.shape(Vq)[1])
    img = np.ones((np.shape(Vq)[0]*xf,yf*np.shape(Vq)[1]))
    for ii in range(0,np.shape(Vq)[0]):
        for jj in range(0,np.shape(Vq)[1]):
            #get indices og img to replace
            xmin = ii*xf;
            xmax = ii*xf + xf;
            ymin = jj*yf;
            ymax = jj*yf + yf;
        
            #if mask[ii,jj] == 0 then replace with matrix of 255s
            if mask[ii,jj] == 0:
                img[xmin:xmax,ymin:ymax] = 255;
            #else mask[ii,jj] =! 0 then replace with correct tile
            else:
                ##compute appropriate tile
                tileN = str(np.int_(np.max((np.min((np.ceil(Vq[ii,jj]/5), 7)), 0))))
                #place tile into img
                img[xmin:xmax,ymin:ymax] = tiles[tileN];

    #add a white border
    img = np.pad(img, (24,24), 'constant', constant_values=255)

    #remove some of the interior white space
    rowcen = np.int_(np.round(np.shape(img)[0]/2));
    colcen = np.int_(np.round(np.shape(img)[1]/2));
    img = np.delete(img, np.s_[np.int_(rowcen-(xf/2)):np.int_(rowcen+(xf/2))], 0)
    img = np.delete(img, np.s_[np.int_(colcen-(yf/2)):np.int_(colcen+(yf/2))], 1)

    plt.figure(figsize=(15,15))

    plt.imshow(img,cmap='gray', vmin=0, vmax=255)

    #draw axes/crosshair: compute key variables
    mx = np.round(np.shape(img)[1]/2)    # horizontal (y axis) centre
    my = np.round(np.shape(img)[0]/2)    # vertical (x axis) centre
    xlims = [yf/2, np.shape(img)[1]-yf/2] # range of x axis (hacked for positioning)
    ylims = [0, np.shape(img)[1]] # range of y axis (continue right to edge)

    #plot axes (cardinal lines)
    plt.plot([mx,mx], ylims, 'k') # plot y axis
    plt.plot(xlims,[my,my] , 'k') # plot x axis
    
    plt.savefig(savestring + '.png')