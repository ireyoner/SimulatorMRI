#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division

def get_Bresenham_line(x1, y1, x2, y2):
    '''Bresenham's line algorithm'''
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y1
    ystep = None
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    if issteep:
        for x in range(x1, x2 + 1):
            points.append((y, x))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
    else:
        for x in range(x1, x2 + 1):
            points.append((x, y))
            error -= deltay
            if error < 0:
                y += ystep
                error += deltax
    return points

def move(point, vector):
    return [point[0] + vector[0], point[1] + vector[1]]

def Radon(Image, SumAlpha, Steps, SensorsNo, SensorsAngle = 30):
    """
    Funkcja zwracająca obraz po transformacji Radona.

    Parameters
    ----------
    Image: 2D/3D array
        obraz do przetworzenia
    SumAlpha: float
        Współczynnik sumowania wartości w przetwarzanym obrazie
    Steps: int
        liczba pozycji emitera (poszczególne pozycje będą oddalone o 360/Steps stopni kątowych)
    SensorsNo : int
        liczba czujników
    SensorsAngle : float (default 180)
        kąt rozwarcia czujników, wyrażony w stopniach kątowych (dla 360 mamy MRI IV generacji).

    Returns
    -------
    Image: 2D/3D array (H,W)
        obraz po przetowrzeniu
        H = Steps (wiersze odpowiadają poszcególnym pozycjom emitera)
        W = SensorsNo (kolumny odpowiadają poszcególnym pozycjom czujników)
        w każdym polu znajduje się liczba odpowiadająca sumie wartości pikseli na linii
            Emiter - Czujnik zwielokrotniona o wartość współczynnika SumAlpha
    """
    import numpy as np
    from math import pi, cos, sin, radians, hypot, ceil, floor, degrees

    R = hypot(len(Image),len(Image[0]))/2

    Center = (len(Image[0])/2,len(Image)/2)

    if Image[0][0].size > 1:
        ret = np.zeros((Steps,SensorsNo,Image[0][0].size))
        I2 = np.zeros((len(Image)%2+ceil(R)*2+2, len(Image[0])%2+ceil(R)*2+2,Image[0][0].size))
    else:
        ret = np.zeros((Steps,SensorsNo))
        I2 = np.zeros((len(Image)%2+ceil(R)*2+2, len(Image[0])%2+ceil(R)*2+2))
    Center2 = (len(I2[0])/2,len(I2)/2)
    I2[Center2[1]-Center[1]:Center2[1]+Center[1],Center2[0]-Center[0]:Center2[0]+Center[0]] = Image
    Center = Center2

    for angle in range(0,Steps):
        alpha = pi*2*angle/Steps
        Emiter = move( Center, ( cos(alpha) * R, sin(alpha) * R ) )
        print('Angle:',round(degrees(alpha),3),'\tEmiter:',Emiter)
        for y, SensorDelta in enumerate(np.linspace(-SensorsAngle/2,SensorsAngle/2, SensorsNo)):
            SensAlpha = alpha+pi+radians(SensorDelta)
            Sensor = move(Center, (cos(SensAlpha)*R, sin(SensAlpha)*R))
            for Piksel in get_Bresenham_line(int(round(Emiter[0])),int(round(Emiter[1])),int(round(Sensor[0])),int(round(Sensor[1]))):
                ret[angle,y] += I2[Piksel[1],Piksel[0]]
    ret *= SumAlpha
    return ret

def mask(Image,Size):
    """
    Funkcja zwracająca obraz po filtracji.

    Parameters
    ----------
    Image: 2D/3D array
        obraz do przetworzenia
    Size: int
        Liczba znaczących liczb w masce
        dla 0 -> [1]
        dla 1 -> [-4*pi**2/1 1. -4*pi**2/1]
        dla 2 -> [-4*pi**2/9 0 -4*pi**2/1 1. -4*pi**2/1 0 -4*pi**2/9]
        ...

    Returns
    -------
    Image: 2D/3D array (H,W)
        obraz po przefiltrowaniu
    """
    import numpy as np
    from skimage.filter.edges import convolve
    from math import pi

    if Image[0][0].size > 1:
        mask = np.zeros((max(Size*4-1,1),max(Size*4-1,1),Image[0][0].size))
    else:
        mask = np.zeros((max(Size*4-1,1),max(Size*4-1,1)))

    numerator = -4/pi**2
    middle = len(mask)//2
    mask[middle][middle] = 1
    for x in range(0,Size):
        pos = (middle - 1) - 2*x
        mask[middle][pos] = mask[middle][-pos-1] = numerator/(middle - pos)**2
    print('Mask: ', mask[middle])
    return convolve(Image, mask)

def backRadon(Image, SumAlpha, Steps, SensorsNo, BackImageWidth, BackImageHeight, SensorsAngle = 180):
    """
    Funkcja zwracająca obraz po odwrotnej transformacie Radona.

    Parameters
    ----------
    Image: 2D/3D array
        obraz do przetworzenia (przetworzony transformatą Radona)
    SumAlpha: float
        Współczynnik sumowania wartości w przetwarzanym obrazie
    Steps: int
        liczba pozycji emitera (poszczególne pozycje będą oddalone o 360/Steps stopni kątowych)
    SensorsNo : int
        liczba czujników
    BackImageWidth : int
        oczekiwana szerokość odtorzonego obrazu
    BackImageHeight : int
        oczekiwana wysokość odtorzonego obrazu
    SensorsAngle : float (default 180)
        kąt rozwarcia czujników, wyrażony w stopniach kątowych (dla 360 mamy MRI IV generacji).

    Returns
    -------
    Image: 2D/3D array (H,W)
        obraz po odtowrzeniu
        H = BackImageHeight
        W = BackImageWidth
    """
    import numpy as np
    from math import pi, cos, sin, radians, hypot, ceil, floor, degrees
    import copy

    R = hypot(BackImageHeight,BackImageWidth)/2

    Center = (BackImageWidth/2,BackImageHeight/2)

    if Image[0][0].size > 1:
        I2 = np.zeros((BackImageHeight%2+ceil(R)*2+2, BackImageWidth%2+ceil(R)*2+2,Image[0][0].size))
    else:
        I2 = np.zeros((BackImageHeight%2+ceil(R)*2+2, BackImageWidth%2+ceil(R)*2+2))

    Center = (len(I2[0])/2,len(I2)/2)

    for angle in range(0,Steps):
        alpha = pi*2*angle/Steps
        Emiter = move( Center, ( cos(alpha) * R, sin(alpha) * R ) )
        print('Angle:',round(degrees(alpha),3),'\tEmiter:',Emiter)
        for y, SensorDelta in enumerate(np.linspace(-SensorsAngle/2,SensorsAngle/2, SensorsNo)):
            SensAlpha = alpha+pi+radians(SensorDelta)
            Sensor = move(Center, (cos(SensAlpha)*R, sin(SensAlpha)*R))
            for Piksel in get_Bresenham_line(int(round(Emiter[0])),int(round(Emiter[1])),int(round(Sensor[0])),int(round(Sensor[1]))):
                I2[Piksel[1],Piksel[0]] += Image[angle,y]
    size = ((len(I2)-BackImageHeight)/2,(len(I2[0])-BackImageWidth)/2)
    I2 = I2[size[0]:size[0]+BackImageHeight,size[1]:size[1]+BackImageWidth]

    return I2


def main(argv):
    import types
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    from skimage.io import imread
    import getopt
    from skimage import data_dir
    import copy
    import os.path

    def normalize(Image,by255=True):
        maxs = [0,0,0]
        output = copy.deepcopy(Image)
        for line in output:
            for elem in line:
                for x,v in enumerate(elem):
                    maxs[x] = max(maxs[x],v)
        if by255:
            maxs = [mx/255 for mx in maxs]
        for line in output:
            for elem in line:
                elem /= maxs
        if by255:
            return output.astype(int)
        else:
            return output

    def RepresentsInt(s):
        try:
            if(int(s)==float(s)):
                return True
            return False
        except ValueError:
            return False

    def RepresentsNumber(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    inputfile = ''
    outputfile = ''
    positions = 360
    sensors = 200
    sumWsp = 1/128
    sensorsAngle = 180
    maskSize = 10

    try:
        opts, args = getopt.getopt(argv,"hi:o:p:s:w:a:m:",["ifile=","ofile=","positions=","sensorsNo=","sumWsp=","sensorsAngle=","maskSize="])
    except getopt.GetoptError:
          print('test.py -i <inputfile> -o <outputfile>')
          sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('test.py \
                    -i <Inputfile> \
                    -o <Outputfile> \
                    -p <No. of Emiter Positions> \
                    -s <No. of Sensors> \
                    -w <Wsp. for sum counting> \
                    -a <Sensors Angle> \
                    -m <Mask Size>\
                    ')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
            if(not (os.path.isfile(inputfile))):
                print('The input file does not exist')
                sys.exit(1)
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-p", "--positions"):
            if(RepresentsInt(arg)):
                positions = int(arg)
                if(positions<1):
                    print('Positions must be an integer >1')
                    sys.exit(1)
            else:
                print('Positions must be an integer')
                sys.exit(1)
        elif opt in ("-s", "--sensorsNo"):
            if(RepresentsInt(arg)):
                sensors = int(arg)
                if(sensors<1):
                    print('Sensors number must be an integer >1')
                    sys.exit(1)
            else:
                print('Sensors must be an integer')
                sys.exit(1)
        elif opt in ("-w", "--sumWsp"):
            if(RepresentsNumber(arg)):
                sumWsp = float(arg)
                if(sumWsp<=0):
                    print('The sum wsp  must be a number greater than zero')
                    sys.exit(1)
            else:
                print('Sum wsp must be a number')
                sys.exit(1)
        elif opt in ("-a", "--sensorsAngle"):
            if(RepresentsInt(arg)):
                sensorsAngle = int(arg)
                if(sensorsAngle<1 or sensorsAngle>360):
                    print('Sensors angle must be an integer in [1:360] range')
                    sys.exit(1)
            else:
                print('Sensors angle must be an integer')
                sys.exit(1)
        elif opt in ("-m", "--maskSize"):
            if(RepresentsInt(maskSize)):
                maskSize = int(arg)
                if(maskSize<1):
                    print('Mask size must be an integer >1')
                    sys.exit(1)
            else:
                print('Sensors angle must be an integer')
                sys.exit(1)

    if inputfile == '':
        print('You must state the input file. \n Usage:   test.pu -i <inputfile> \n Use -h for help and list of options')
        sys.exit(1)
    else:
        Image = imread(inputfile)

    radomImage = Radon(Image,sumWsp,positions,sensors,sensorsAngle)
    maskedImage = mask(radomImage,maskSize)
    backRImage = backRadon(maskedImage,sumWsp,positions,sensors,len(Image[0]),len(Image),sensorsAngle)

    if Image[0][0].size > 1:
        radomImage = normalize(radomImage)
        maskedImage = normalize(maskedImage,False)
        backRImage = normalize(backRImage,False)
        colormap = cm.get_cmap()
    else:
        colormap = cm.Greys_r

    line_figurs = 4

    fig = plt.figure()
    ax1 = fig.add_subplot(1, line_figurs, 1)
    ax4 = fig.add_subplot(2, line_figurs, line_figurs)
    ax5 = fig.add_subplot(2, line_figurs, line_figurs*2)

    ax1.set_title("Original Image")
    ax1.imshow(Image, cmap = colormap)
    ax1.axis('off')

    if Image[0][0].size > 1:
        for x in range(0,3):
            ax2 = fig.add_subplot(4, line_figurs, line_figurs+2+x*line_figurs)
            ax3 = fig.add_subplot(4, line_figurs, line_figurs+3+x*line_figurs)
            ax2.imshow(radomImage[:,:,x], cmap = cm.Greys_r)
            ax2.set_ylabel('Emiter position')
            ax3.imshow(maskedImage[:,:,x], cmap = cm.Greys_r)
            ax3.set_ylabel('Emiter position')
        ax2.set_xlabel('Sensor')
        ax3.set_xlabel('Sensor')
        ax2 = fig.add_subplot(4, line_figurs, 2)
        ax3 = fig.add_subplot(4, line_figurs, 3)
        ax2.set_title("Radon Transform")
        ax2.imshow(radomImage, cmap = colormap)
        ax2.set_ylabel('Emiter position')
        ax3.set_title("Masked")
        ax3.imshow(maskedImage, cmap = colormap)
        ax3.set_ylabel('Emiter position')

    else:
        ax2 = fig.add_subplot(1, line_figurs, 2)
        ax3 = fig.add_subplot(1, line_figurs, 3)
        ax2.set_title("Radon Transform")
        ax2.imshow(radomImage, cmap = colormap)
        ax2.set_xlabel('Sensor')
        ax2.set_ylabel('Emiter position')
        ax3.set_title("Masked")
        ax3.imshow(maskedImage, cmap = colormap)
        ax3.set_xlabel('Sensor')
        ax3.set_ylabel('Emiter position')

    ax4.set_title("Back Radon Transform")
    ax4.imshow(backRImage, cmap = colormap)
    ax4.axis('off')

    ax5.set_title("Absolute Error")
    ax5.imshow(abs(Image - backRImage), cmap = colormap)
    ax5.axis('off')

    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    if outputfile != '':
        fig.set_size_inches(18.5,10.5)
        plt.savefig(outputfile+'.png', dpi=100)
    else:
        plt.show()

    return


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
